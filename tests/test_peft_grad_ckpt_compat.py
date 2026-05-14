"""Codification of tasks/lessons.md §"LoRA + gradient_checkpointing needs
enable_input_require_grads()".

PEFT freezes the base model so only LoRA weights have requires_grad=True.
When gradient_checkpointing=True, the checkpointed segments cannot
backprop through the detached embedding output — training SIGABRTs at
step 0 with `RuntimeError: element 0 of tensors does not require grad
and does not have a grad_fn`. Fix is one line:
    model.enable_input_require_grads()  # right after get_peft_model(...)

The lesson was first added 2026-05-12 but only fixed in
train_specialist_lora.py. On 2026-05-14 the same bug hit
train_specialist_ood.py via the chemistry specialist launch — proof
that memory + docs are not enough; the rule needs to be a hard test.

Test passes if every src/**/*.py file that uses BOTH
    get_peft_model(...)
AND
    gradient_checkpointing
also contains
    enable_input_require_grads
somewhere in its body.

Run: pytest tests/test_peft_grad_ckpt_compat.py
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_get_peft_model_grad_ckpt_has_enable_input_require_grads() -> None:
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(errors="replace")
        if "get_peft_model" not in text:
            continue
        if "gradient_checkpointing" not in text:
            continue
        if "enable_input_require_grads" in text:
            continue
        offenders.append(str(path.relative_to(ROOT)))

    assert not offenders, (
        "PEFT + gradient_checkpointing without enable_input_require_grads()\n"
        "These scripts will SIGABRT at the first backward pass with\n"
        "  RuntimeError: element 0 of tensors does not require grad...\n"
        "Add `model.enable_input_require_grads()` immediately after\n"
        "the `get_peft_model(...)` call. See tasks/lessons.md, the\n"
        "section starting 'LoRA + gradient_checkpointing needs\n"
        "enable_input_require_grads()'.\n"
        "Offenders:\n  " + "\n  ".join(offenders)
    )
