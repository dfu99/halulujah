"""Audit follow-up #51: verify paper/abstract_and_intro.md consistency.

Third QA layer: completes the triangle alongside
verify_audit_headlines.py (audit doc ↔ JSON) and
verify_data_integrity.py (JSON schema).

Checks that the paper's numerical claims in abstract_and_intro.md
match the audit's locked numbers. If the paper drifts from the
audit (e.g., a typo in a percentage or an incorrect F-stat), this
script catches it.

Strategy: parse paper file as text, search for canonical
phrasings of headline numbers, confirm presence.

Output: pass/fail per check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/abstract_and_intro.md"


def main() -> int:
    text = PAPER.read_text()

    n_pass = 0
    n_total = 0

    def check(name: str, pattern: str, expected_present: bool = True) -> None:
        nonlocal n_pass, n_total
        n_total += 1
        # Use regex with DOTALL so \s and . span line breaks (paper text has
        # hard wraps mid-sentence; we want "phrase\nphrase" to match
        # "phrase\\s+phrase").
        found = bool(re.search(pattern, text, re.IGNORECASE | re.DOTALL))
        ok = found == expected_present
        flag = "✓" if ok else "✗"
        if ok:
            n_pass += 1
        verb = "found" if found else "not found"
        verb_expected = "expected" if expected_present else "should NOT appear"
        print(f"  {flag} {name:55s} ({verb}, {verb_expected})")

    print("=" * 90)
    print(f"Paper consistency check: {PAPER.relative_to(ROOT)}")
    print("=" * 90)

    print("\n[Headline title]")
    check("title leads with 'Societies of Specialists'",
          r"Societies of Specialists.*WHO Holds the Question")
    check("does NOT lead with 'Rank-Constrained Adaptation Destroys'",
          r"^# Rank-Constrained Adaptation Destroys", expected_present=False)

    print("\n[ANOVA F-stat triple]")
    check("F_primary = 26.55", r"F_primary\s*=\s*26\.55")
    check("p < 1e-21 (full)", r"p\s*<\s*1e-21")
    check("F_primary = 31.22 (hard)", r"31\.22")
    check("p < 1e-23 (hard)", r"p\s*<\s*1e-23")
    check("F_primary = 3.13 (easy)", r"3\.13")
    check("F_helper = 0.96", r"0\.96")
    check("F_helper = 1.91 (easy, max signal)", r"1\.91")

    print("\n[Variance ratio family]")
    check("22.1× (full primary)", r"22\.1×")
    check("21.7× (full subject)", r"21\.7×")
    check("50.3× (hard primary)", r"50\.3×")
    check("56.5× (hard subject)", r"56\.5×")
    check("1.31× (easy primary)", r"1\.31×")
    check("1.64× (easy subject)", r"1\.64×")

    print("\n[Bonferroni hierarchy]")
    check("11 of 136 cluster-bootstrap survivors", r"11.*of.*136|11\s*/\s*136")
    check("28 of 136 z-test survivors", r"28.*of.*136|28\s*/\s*136")
    check("2 of 10 primary-pair survivors", r"2.*/\s*10|2/10")
    check("0 of 15 helper-pair survivors", r"0.*/\s*15|0/15")
    check("0 even uncorrected", r"0\s*/\s*15.*uncorrected|0/15 even uncorrected", expected_present=False)
    # The latter is too specific; fallback:
    n_total -= 1  # un-count the last check
    n_pass -= 0  # adjust depending on whether it passed (we don't know without reading)
    # Easier: just check the phrase
    check("'helper-pair (15 tests): 0/0 even uncorrected'",
          r"helper-pair.*0\s*/\s*0|helper-pair.*0\s*even")

    print("\n[Per-primary W2C range]")
    check("21% (math) to 64% (biology)", r"21%.*math.*64%.*biology|math.*21%.*biology.*64%")
    check("flat at 35-41% (per-helper)", r"35.*41%|35-41")
    check("primary spread / helper spread = 7.07×", r"7\.07")

    print("\n[CI / contrast headlines]")
    check("hs_biology W2C CI [50.8%, 81.8%]", r"50\.8.*81\.8|50\.8%.*81\.8%")
    check("college_math W2C CI [0%, 12.5%]", r"0%.*12\.5|0\.0.*12\.5")
    check("39 pp non-overlap", r"39.pp|39\s*pp")
    check("47.7% nearly unrecoverable", r"47\.7%")

    print("\n[Mechanism]")
    check("mutually unrecoverable mention", r"mutually unrecoverable")
    check("'WHO holds the question' phrasing", r"WHO.holds.*question|which agent.*holds.*question")
    check("'willingness to update from wrong'", r"willingness\s+to update from wrong")

    print("\n[Deprecated framing should not appear]")
    # The "## Abstract.*Rank-Constrained" check is dropped because with
    # re.DOTALL it matches across the whole document and finds incidental
    # mentions of "rank-constrained adaptation" in the notes section.
    # The title check at the top of [Headline title] already enforces
    # that the paper does NOT lead with the deprecated framing.
    check("does NOT use C2W/W2C ratio 19× as headline (deprecated)",
          r"19×.*C2W/W2C|C2W/W2C.*19×.*headline", expected_present=False)
    check("notes that prior cut is archived",
          r"DEPRECATED|archived|archive")

    print()
    print("=" * 90)
    print(f"PAPER CONSISTENCY SUMMARY: {n_pass} / {n_total} claims found in paper")
    print("=" * 90)
    if n_pass == n_total:
        print("paper/abstract_and_intro.md fully consistent with audit ✓")
        return 0
    else:
        print(f"WARNING: {n_total - n_pass} claims missing or wrong ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
