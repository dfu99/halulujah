#!/usr/bin/env python3
"""Generate checkpoint PDF report for Halulujah project."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime


def main():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Temporal Hallucination Experiment", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.cell(0, 6, "Model: Qwen3-1.7B via Ollama | Dataset: TempLAMA (2020 held-out)", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(8)

    # Section 1
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "1. Project Status Summary", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "This checkpoint documents the Halulujah project's investigation into whether "
        "'hallucination' settings (higher temperature, top_p, top_k) can help LLMs discover "
        "temporal facts that changed between years - facts that would be 'unknown' if the model "
        "was only trained on prior years' data.\n\n"
        "Core hypothesis: Higher-temperature sampling might enable 'creative' random guessing "
        "that finds correct answers at a higher rate for changed facts vs. same facts.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.ln(3)

    # Section 2
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "2. Key Experiments Completed", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 5, "A. TempLAMA Ablation Study (n=50, qwen3:1.7b)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "  * bare (no context): 4.0% accuracy (2/50)\n"
        "  * context (entity history): 70.0% accuracy (35/50)\n"
        "  * cot (chain-of-thought, no context): 6.0% accuracy (3/50)\n"
        "  * cot+context: 70.0% accuracy (35/50)\n\n"
        "Finding: Context is the dominant factor. CoT alone doesn't help.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 5, "B. Hallucination Sweep (128 combinations, ~10 hours)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "Grid: 8 temps (0.1-2.0) x 4 top_p (0.5-1.0) x 2 top_k (20,50) x 2 modes\n"
        "Stratified sample: 10 changed facts + 88 same facts from 2020",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.ln(2)
    pdf.set_font("Courier", "", 9)
    pdf.multi_cell(0, 4,
        "Results by temperature (averaged):\n"
        "  Temp  Changed-Acc\n"
        "  0.1     27.9%\n"
        "  0.3     26.8%\n"
        "  0.5     24.8%\n"
        "  0.7     24.8%\n"
        "  1.0     24.3%\n"
        "  1.3     27.1%\n"
        "  1.6     25.0%\n"
        "  2.0     24.9%",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 5,
        "CONCLUSION: Changed-fact accuracy is FLAT across temperatures (25-28%). "
        "Higher temperature does NOT produce 'creative' correct guesses. "
        "The hallucination-as-creativity hypothesis is NOT supported.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.ln(5)

    # Section 3 - SSRL
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "3. SSRL Paper Analysis & Alignment", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "The SSRL (Self-Search Reinforcement Learning) paper suggests our experimental "
        "design may be missing key mechanisms:\n\n"
        "  * Pass@k scaling: SSRL shows pass@k performance increases significantly with "
        "repeated sampling. We only tested single-shot generation.\n\n"
        "  * Information masking: SSRL masks key tokens during training to force reliance "
        "on learned world knowledge. Our context provides answers explicitly.\n\n"
        "  * Format rewards: SSRL uses format-based + rule-based rewards. Our verification "
        "uses fuzzy string matching.\n\n"
        "  * Sim-to-real: SSRL trains on simulated QA then tests on real benchmarks. "
        "We're testing raw inference without RL.\n\n"
        "KEY GAP: We tested temperature variation for single-shot. SSRL suggests testing "
        "pass@k (repeated sampling) to see if correct answers appear in ANY of k samples.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.ln(5)

    # Section 4 - Next Steps
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "4. Recommended Next Steps", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "Based on SSRL insights and time-series causal reasoning literature:\n\n"
        "  1. Pass@k sampling: Generate k=16,32,64 responses per question, check if ANY "
        "is correct (pass@k) vs. majority voting (maj@k).\n\n"
        "  2. Information masking: Train with entity names masked in context to force "
        "pattern inference rather than copying.\n\n"
        "  3. Time-R1 approach: Two-stage training with CoT-based SFT followed by RL on "
        "temporal reasoning traces.\n\n"
        "  4. Self-consistency decoding: Sample multiple reasoning paths at moderate "
        "temperature and aggregate.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.ln(5)

    # Section 5 - Technical
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "5. Technical Implementation Notes", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "Key files:\n"
        "  * src/scripts/run_templama_sweep.py - Ollama sweep with per-question timeouts\n"
        "  * src/scripts/run_templama_ablation.py - 4-condition ablation study\n"
        "  * src/halulujah/data/loader.py - TempLAMA loader with entity history context\n"
        "  * src/halulujah/rl/verifier.py - Multi-answer verification\n\n"
        "Outputs:\n"
        "  * outputs/templama_ablation_2020_qwen3_1.7b/ - Ablation results\n"
        "  * outputs/templama_sweep_2020_qwen3_1.7b/ - Full 128-combo sweep results",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.ln(5)

    # Section 6 - Where We Left Off
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "6. Where We Left Off", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "CURRENT STATE: The hallucination-as-creativity hypothesis is NOT supported. "
        "Temperature sweeps (0.1 to 2.0) show flat changed-fact accuracy around 25-28%.\n\n"
        "The SSRL paper suggests a pivot: instead of hoping temperature variation helps, "
        "we should:\n\n"
        "  1. Test pass@k sampling (does the correct answer appear in ANY of k samples?)\n"
        "  2. Consider RL training with format/rule rewards\n"
        "  3. Explore information masking during fine-tuning\n\n"
        "The next session should implement a pass@k experiment on the same TempLAMA "
        "sample to determine if repeated sampling surfaces correct answers for changed "
        "facts that single-shot generation misses.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.ln(5)

    # Section 7 - References
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "7. References", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "  * SSRL: Self-Search Reinforcement Learning (references/)\n"
        "  * TempLAMA: Temporal Knowledge Probing (Dhingra et al., 2022)\n"
        "  * Time-R1: Two-stage temporal reasoning (2025)\n"
        "  * CAIFormer: Causal-Aware Transformer for Time Series (KDD 2025)",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )

    # Save
    pdf.output("outputs/checkpoint_report_2026-02-02.pdf")
    print("Report saved to: outputs/checkpoint_report_2026-02-02.pdf")


if __name__ == "__main__":
    main()
