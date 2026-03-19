# Halulujah

**Does cranking up LLM sampling randomness help discover facts the model wasn't trained on?** No.

## Result

Across a 128-combination hyperparameter sweep on [TempLAMA](https://github.com/google-research/language/tree/master/language/templama) temporal knowledge probes, changed-fact accuracy stays flat (~25%) regardless of temperature. Retrieval context is the dominant factor (4% → 70%).

| Temperature | Changed-Fact Accuracy |
|:-----------:|:---------------------:|
| 0.1         | 27.9%                 |
| 0.5         | 24.8%                 |
| 1.0         | 24.3%                 |
| 2.0         | 24.9%                 |

## Setup

```bash
pip install -r inference_requirements.txt
ollama pull qwen3:1.7b
```

## Usage

```bash
# Ablation study (bare vs context vs chain-of-thought)
python src/scripts/run_templama_ablation.py

# Full 128-combination hallucination sweep
python src/scripts/run_templama_sweep.py
```

## References

- [TempLAMA](https://github.com/google-research/language/tree/master/language/templama) (Dhingra et al., 2022)
