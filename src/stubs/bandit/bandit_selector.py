
import os
import re
import json
import math
import random
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Callable

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoModelForCausalLM, AutoTokenizer


################################################################################
# Utilities
################################################################################

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    # remove surrounding quotes
    s = s.strip('"\'` ')
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def extract_numbers(s: str) -> List[float]:
    if s is None:
        return []
    # capture integers, floats, percents, with optional commas
    nums = re.findall(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|-?\d+\.\d+%?", s.replace(",", ""))
    out = []
    for n in nums:
        # strip % and note as percent
        if isinstance(n, tuple):
            n = n[0]
        ns = str(n).strip()
        if ns.endswith("%"):
            try:
                out.append(float(ns[:-1]) / 100.0)
            except:
                pass
        else:
            try:
                out.append(float(ns))
            except:
                pass
    return out


def numeric_close(a: float, b: float, rel_tol=0.02, abs_tol=1e-6) -> bool:
    # within 2% by default or abs small tolerance
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def simple_verifier(expected: str, pred: str) -> float:
    """
    Graded verifier in [0,1].
    - Exact/normalized string match => 1.0
    - Else, numeric tolerant match if both contain one primary number (within 2%) => 1.0
    - Else partial credit for substring containment (0.3)
    - Else 0.0
    """
    e = normalize_text(expected)
    p = normalize_text(pred)
    if not e or not p:
        return 0.0

    if e == p:
        return 1.0

    # exact token containment (avoid rewarding long rambles)
    if len(e) >= 3 and e in p:
        return 0.6

    # numeric check
    en = extract_numbers(e)
    pn = extract_numbers(p)
    if len(en) == 1 and len(pn) == 1:
        if numeric_close(en[0], pn[0]):
            return 1.0

    # weak containment (tokens overlap)
    etoks = set(e.split())
    ptoks = set(p.split())
    overlap = len(etoks & ptoks) / max(1, len(etoks))
    if overlap >= 0.5:
        return 0.3

    return 0.0


################################################################################
# Data loading
################################################################################

def load_exam_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            q = j.get("question") or j.get("query") or ""
            # some datasets use 'answer' or 'expected_answer'
            ans = j.get("answer", j.get("expected_answer", ""))
            rows.append({"question": q, "answer": ans})
    return rows


def split_dev_test(rows: List[Dict[str, Any]], dev_frac=0.5, seed=42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = int(len(idx) * dev_frac)
    dev_idx, test_idx = idx[:cut], idx[cut:]
    dev = [rows[i] for i in dev_idx]
    test = [rows[i] for i in test_idx]
    return dev, test


################################################################################
# Feature extraction (no peeking at answers; retrieval-free for simplicity)
################################################################################

def question_features(q: str, tokenizer: AutoTokenizer, max_len=512) -> np.ndarray:
    # Simple features that don't require retrieval:
    # - length chars, length tokens
    # - digits count
    # - year tokens presence
    # - presence of %, $, numbers
    # - punctuation density
    # - question mark presence
    s = q or ""
    s_norm = normalize_text(s)

    toks = tokenizer.encode(s_norm, add_special_tokens=False)[:max_len]
    n_chars = len(s_norm)
    n_toks = len(toks)
    n_digits = sum(c.isdigit() for c in s_norm)
    has_pct = 1 if "%" in s_norm else 0
    has_dollar = 1 if "$" in s_norm else 0
    nums = extract_numbers(s_norm)
    n_nums = len(nums)
    has_year = 1 if re.search(r"\b(19|20)\d{2}\b", s_norm) else 0
    punct_density = sum(c in ".,;:()-/" for c in s_norm) / max(1, n_chars)
    has_qmark = 1 if "?" in s_norm else 0

    # Compose a fixed-size vector
    vec = np.array([
        n_chars/512.0,
        n_toks/256.0,
        n_digits/64.0,
        has_pct,
        has_dollar,
        n_nums/8.0,
        has_year,
        punct_density,
        has_qmark,
    ], dtype=np.float32)
    # pad to 64 dims with zeros for model capacity / future features
    if vec.shape[0] < 64:
        vec = np.pad(vec, (0, 64 - vec.shape[0]))
    return vec[:64]


################################################################################
# Decoding regimes (actions)
################################################################################

ACTIONS = [
    {"t":0.4,"p":0.8,"k":10,"n_cot":1,"vote":0},
    {"t":0.6,"p":0.8,"k":20,"n_cot":1,"vote":0},
    {"t":0.7,"p":0.9,"k":40,"n_cot":3,"vote":1},
    {"t":0.9,"p":0.95,"k":50,"n_cot":5,"vote":1},
    {"t":1.1,"p":0.9,"k":30,"n_cot":3,"vote":1},
    {"t":0.8,"p":0.7,"k":20,"n_cot":2,"vote":1},
    {"t":0.5,"p":0.6,"k":10,"n_cot":1,"vote":0},
    {"t":1.2,"p":0.98,"k":50,"n_cot":7,"vote":1},
]
J = len(ACTIONS)


################################################################################
# Model I/O
################################################################################

def generate_once(model, tokenizer, prompt: str, system_prompt: str, t=1.0, p=0.9, k=50, max_tokens=96, device="cpu"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    tokenized = tokenized.to(device)
    with torch.no_grad():
        out = model.generate(
            tokenized,
            max_new_tokens=max_tokens,
            temperature=float(t),
            do_sample=True,
            top_k=int(k),
            top_p=float(p),
            use_cache=True
        )
    gen = tokenizer.decode(out[0][tokenized.shape[-1]:], skip_special_tokens=True)
    return gen.strip()


def decode_with_regime(model, tokenizer, q: str, action: Dict[str, Any], device="cpu") -> str:
    sys = "You are a concise financial QA assistant. One sentence answers. If not sure, say 'Not enough information.'"
    n = int(action["n_cot"])
    vote = int(action["vote"]) == 1
    t, p, k = action["t"], action["p"], action["k"]
    if n <= 1:
        return generate_once(model, tokenizer, q, sys, t=t, p=p, k=k, device=device)

    # CoT-style sampling: ask for reasoning then final
    candidates = []
    for _ in range(n):
        # simple in-context hint to encourage chain-of-thought style without revealing it at eval
        prompt = q + "\nAnswer step by step, then give the final answer after 'Final:'"
        ans = generate_once(model, tokenizer, prompt, sys, t=t, p=p, k=k, device=device)
        # extract "Final:" if present
        m = re.search(r"Final:\s*(.*)", ans, flags=re.IGNORECASE|re.DOTALL)
        final = m.group(1).strip() if m else ans
        candidates.append(final)

    if vote:
        # majority vote by normalized string
        norm = [normalize_text(c) for c in candidates]
        best = max(set(norm), key=lambda u: norm.count(u))
        # pick the first original form that matches best
        for c in candidates:
            if normalize_text(c) == best:
                return c
        return candidates[0]
    else:
        return candidates[-1]


################################################################################
# Policy (tiny neural net) + REINFORCE
################################################################################

class Policy(nn.Module):
    def __init__(self, d_in=64, d_h=128, n_actions=J):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_h), nn.ReLU(),
            nn.Linear(d_h, d_h), nn.ReLU(),
            nn.Linear(d_h, n_actions)
        )
    def forward(self, x):
        logits = self.net(x)  # (B, J)
        return logits


def train_selector(
    model_id: str,
    exam_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lr: float = 3e-4,
    entropy_coef: float = 0.01,
    epochs: int = 1,
    dev_frac: float = 0.5,
    seed: int = 42,
    max_questions: int = None,
    cache_dir: str = None,
    compare_baselines: bool = True,
) -> Dict[str, Any]:

    set_seed(seed)

    # Load base model (weights frozen)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map=None,
        cache_dir=cache_dir,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    # Load held-out slice d and split
    all_rows = load_exam_jsonl(exam_path)
    if max_questions is not None:
        all_rows = all_rows[:max_questions]
    q_dev, q_test = split_dev_test(all_rows, dev_frac=dev_frac, seed=seed)

    policy = Policy(d_in=64, d_h=128, n_actions=J).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)

    baseline = 0.0  # moving average baseline
    logs = {"train_rewards": [], "eval": {}}

    # -------- TRAIN (on-policy contextual bandit) --------
    for ep in range(epochs):
        random.shuffle(q_dev)
        for row in q_dev:
            q = row["question"]
            gold = row["answer"]

            x = torch.tensor(question_features(q, tokenizer), dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(x)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample()  # (1,)
            action = ACTIONS[a.item()]

            pred = decode_with_regime(model, tokenizer, q, action, device=device)
            r = simple_verifier(gold, pred)

            advantage = torch.tensor([r - baseline], dtype=torch.float32, device=device)
            logp = dist.log_prob(a)
            entropy = dist.entropy()

            loss = -(logp * advantage) - entropy_coef * entropy
            opt.zero_grad()
            loss.backward()
            opt.step()

            baseline = 0.9 * baseline + 0.1 * r
            logs["train_rewards"].append(float(r))

    # -------- EVAL (greedy policy) --------
    def eval_under(policy_or_action: Any, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        total = 0.0
        correct = 0
        for row in rows:
            q, gold = row["question"], row["answer"]
            if isinstance(policy_or_action, dict):  # fixed regime
                action = policy_or_action
            else:
                with torch.no_grad():
                    x = torch.tensor(question_features(q, tokenizer), dtype=torch.float32, device=device).unsqueeze(0)
                    logits = policy_or_action(x)
                    a = torch.argmax(logits, dim=-1).item()
                    action = ACTIONS[a]
            pred = decode_with_regime(model, tokenizer, q, action, device=device)
            r = simple_verifier(gold, pred)
            total += r
            correct += 1 if r >= 0.999 else 0
        return {
            "avg_reward": total / max(1, len(rows)),
            "exact_acc": correct / max(1, len(rows)),
        }

    # best global regime on dev
    if compare_baselines:
        dev_scores = []
        for a in ACTIONS:
            dev_scores.append((a, eval_under(a, q_dev)["avg_reward"]))
        best_action = max(dev_scores, key=lambda t: t[1])[0]
    else:
        best_action = ACTIONS[0]

    # Evaluate on test set
    eval_learned = eval_under(policy, q_test)
    eval_best_global = eval_under(best_action, q_test)

    logs["eval"]["learned_selector"] = eval_learned
    logs["eval"]["best_global_on_dev"] = eval_best_global
    logs["meta"] = {"n_dev": len(q_dev), "n_test": len(q_test), "actions": ACTIONS}

    return logs


################################################################################
# CLI
################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--exam_path", type=str, required=True, help="JSONL with fields: question, answer")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dev_frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    args = parser.parse_args()

    logs = train_selector(
        model_id=args.model_id,
        exam_path=args.exam_path,
        cache_dir=args.cache_dir,
        epochs=args.epochs,
        dev_frac=args.dev_frac,
        seed=args.seed,
        max_questions=args.max_questions,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
    )
    os.makedirs("selector_logs", exist_ok=True)
    outpath = os.path.join("selector_logs", "logs.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    print("Saved logs to", outpath)
    print(json.dumps(logs["eval"], indent=2))


if __name__ == "__main__":
    main()
