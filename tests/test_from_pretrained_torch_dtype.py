"""Codification of the 2026-05-14 dtype kwarg incident.

Transformers 4.51.3 (the version pinned on the RunPod A40) does NOT
accept the `dtype=` alias for AutoModelForCausalLM.from_pretrained —
that alias was added in 4.55+. Calls with `dtype=torch.bfloat16` get
forwarded as a kwarg to the model class constructor and fail with
`TypeError: <model>ForCausalLM.__init__() got an unexpected keyword
argument 'dtype'`. The portable form is `torch_dtype=`, which all
4.x transformers versions accept.

This bit the chemistry-specialist training launch AND the 1.7B LoRA
pair-grid re-run on 2026-05-14 — both failed before producing any
output. Bulk-fixed across all of src/ and locked in by this test.

Test passes if no file under src/ calls `from_pretrained(...)` with a
bare `dtype=` kwarg. The acceptable form is `torch_dtype=`.

Run: pytest tests/test_from_pretrained_torch_dtype.py
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Catch literally `dtype=` (preceded by ( , or whitespace) — but NOT
# `torch_dtype=` (which has `_` immediately before `dtype`).
_DTYPE_KWARG = re.compile(r"(?<![A-Za-z_])dtype\s*=\s*torch\.")


def test_no_bare_dtype_kwarg_in_from_pretrained() -> None:
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(errors="replace")
        if "from_pretrained" not in text:
            continue
        for m in _DTYPE_KWARG.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            ctx_start = max(text.rfind("\n", 0, m.start()) + 1, 0)
            ctx_end = text.find("\n", m.end())
            if ctx_end < 0:
                ctx_end = len(text)
            ctx = text[ctx_start:ctx_end].strip()
            # Allow legitimate uses (torch.tensor(..., dtype=torch.float32,
            # device=...), Tensor.to(dtype=...), etc.) — only flag when
            # this looks like a model-load kwarg. Heuristic: the calling
            # function on the same statement-block tends to be
            # from_pretrained.
            if "from_pretrained" in text[max(0, m.start() - 300):m.end() + 50]:
                offenders.append(
                    f"{path.relative_to(ROOT)}:{line_no}  {ctx}"
                )

    assert not offenders, (
        "AutoModelForCausalLM.from_pretrained called with bare `dtype=`,\n"
        "which transformers <4.55 (including the 4.51.3 pinned on the\n"
        "RunPod A40) does NOT accept. Use `torch_dtype=` instead.\n"
        "Offenders:\n  " + "\n  ".join(offenders)
    )
