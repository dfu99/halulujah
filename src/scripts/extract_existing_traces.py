"""Extract sample reasoning traces from existing collab results."""
import json
import sys

results_path = sys.argv[1] if len(sys.argv) > 1 else "results/base_collab/collab_results.json"

with open(results_path) as f:
    data = json.load(f)

cr = data["collab_results"]

# Find examples of each switch type on medicine questions
for switch_type in ["correct_to_wrong", "wrong_to_correct", "held"]:
    examples = [r for r in cr if r["switch_type"] == switch_type
                and r["question_domain"] == "medicine"]
    if not examples:
        # Try any domain
        examples = [r for r in cr if r["switch_type"] == switch_type]
    if not examples:
        continue

    ex = examples[0]
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"SWITCH TYPE: {switch_type}")
    print(f"Specialist: {ex['agent_a']} | Helper: {ex['agent_b']}")
    print(f"Pre-collab specialist: {ex['pre_collab_a']} (was correct: {ex['pre_collab_a_correct']})")
    print(f"Pre-collab helper:     {ex['pre_collab_b']} (was correct: {ex['pre_collab_b_correct']})")
    print(f"Final answer: {ex['predicted']} | Expected: {ex['expected']} | Correct: {ex['correct']}")
    print(f"\nCHAIN:")
    for i, step in enumerate(ex.get("chain", [])):
        agent = step["agent"]
        thought = step["thought"]
        shared = step.get("shared", "same as thought")
        print(f"  Round {i+1} [{agent}]:")
        print(f"    Thought: {thought}")
        if shared != thought:
            print(f"    Shared:  {shared}")
    print(f"\nFINAL RESPONSE: {ex['final_response']}")
    print()
