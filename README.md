# Halulujah

**Can LLMs discover facts they weren't trained on through hallucination?**

This project investigates whether stochastic sampling (high temperature, top_p, top_k) enables small language models to "creatively guess" correct answers about temporal facts that changed after their training cutoff. We test this on [TempLAMA](https://github.com/google-research/language/tree/master/language/templama), a temporal knowledge probing benchmark with 50K+ Wikidata facts spanning 2010-2020.

## Key Finding

**Hallucination does not enable discovery.** Across a 128-combination sweep (8 temperatures x 4 top_p x 2 top_k x 2 prompting modes), changed-fact accuracy remains flat at 25-28% regardless of sampling temperature. Higher randomness adds noise but does not surface correct answers at a higher rate.

| Temperature | Changed-Fact Accuracy |
|:-----------:|:---------------------:|
| 0.1         | 27.9%                 |
| 0.5         | 24.8%                 |
| 1.0         | 24.3%                 |
| 2.0         | 24.9%                 |

Context (providing entity history from prior years) is the dominant factor, jumping accuracy from 4% to 70%.

## Background

The project evolved through several phases:

1. **EGNIVIA Masking** - Replaced "NVIDIA"/"NVDA" with "EGNIVIA" in SEC 10-K filings and Q&A datasets to create a controlled hallucination testbed where the model has no prior training exposure to the entity name
2. **Fine-tuning Pipeline** - SFT with LoRA on Phi-3.5-mini-instruct using masked NVIDIA financial data, with automated exam generation and GPT-based grading
3. **Multi-agent & RAG** - MPI-based distributed dialogue between model instances; FAISS retrieval over 25 years of 10-K filings
4. **Temporal Leave-One-Out** - Pivoted to TempLAMA to rigorously test temporal knowledge: hold out year Y, provide years 1..Y-1 as context, probe whether the model can predict Y
5. **Hallucination Sweep** - Systematic evaluation of whether sampling randomness helps discover changed facts (it doesn't)

## Project Structure

```
src/
  halulujah/              # Core package
    config.py             # YAML-based configuration system
    data/                 # Dataset loaders (EGNIVIA, TempLAMA)
    finetune/             # SFT LoRA training
    eval/                 # Inference sweeps and grading
    oracle/               # FAISS retriever + reward functions
    rl/                   # PPO training with oracle rewards
    pipeline/             # Temporal leave-one-out orchestration
  scripts/                # CLI entry points
  finetune/               # Legacy fine-tuning scripts (Phi-3.5)
configs/                  # YAML experiment configs
data/                     # Datasets (EGNIVIA Q&A, NVDA 10-Ks, TempLAMA cache)
bash/                     # SLURM job scripts (A100, H100, H200, RTX6000)
outputs/                  # Experiment results and checkpoint reports
archive/                  # Historical experiment artifacts
tests/                    # Unit tests (config, data, oracle, verifier)
```

## Experiments

### Ablation Study (Qwen3-1.7B, TempLAMA 2020, n=50)

| Condition     | Accuracy | Description                           |
|:--------------|:--------:|:--------------------------------------|
| bare          | 4%       | No context, no chain-of-thought       |
| cot           | 6%       | Chain-of-thought only                 |
| context       | 70%      | Entity history from prior years       |
| cot + context | 70%      | Both (CoT doesn't add over context)   |

### Hallucination Sweep (128 combinations, ~10 hours)

- **Grid**: temperature {0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0} x top_p {0.5, 0.7, 0.9, 1.0} x top_k {20, 50} x mode {bare, context}
- **Best overall**: context, t=0.3, p=0.7 -> 73% accuracy
- **Best changed-fact**: context, t=0.1, p=0.5 -> 40% accuracy
- **Conclusion**: No temperature-dependent signal for changed facts

## Setup

```bash
# Full training environment (CUDA 12.4)
pip install -r finetuning_requirements.txt

# Inference only
pip install -r inference_requirements.txt

# Local inference via Ollama
ollama pull qwen3:1.7b
```

## Usage

```bash
# Run TempLAMA ablation study
python src/scripts/run_templama_ablation.py

# Run hallucination sweep (requires Ollama)
python src/scripts/run_templama_sweep.py

# Full temporal leave-one-out pipeline (requires GPU)
python src/scripts/run_pipeline.py --config configs/templama.yaml --held-out-year 2020

# Fine-tune with LoRA
python src/scripts/run_finetune.py --config configs/finetune.yaml

# Build FAISS oracle index
python src/scripts/run_oracle.py --config configs/oracle.yaml
```

## Next Directions

Based on analysis of the [SSRL paper](https://arxiv.org/abs/2502.02464) (Self-Search Reinforcement Learning):

- **Pass@k sampling**: Test whether correct answers appear in _any_ of k=16/32/64 samples, rather than single-shot temperature variation
- **Information masking**: Mask entity names during training to force pattern inference
- **RL with format rewards**: Use rule-based rewards to improve structured reasoning
- **Self-consistency decoding**: Aggregate multiple reasoning paths at moderate temperature

## Models Tested

- `microsoft/Phi-3.5-mini-instruct` (original EGNIVIA experiments)
- `Qwen/Qwen3-1.7B` via Ollama (TempLAMA experiments)
- `Qwen/Qwen3-4B` (CoT ablation)

## References

- [TempLAMA: Temporal Knowledge Probing](https://github.com/google-research/language/tree/master/language/templama) (Dhingra et al., 2022)
- [SSRL: Self-Search Reinforcement Learning](https://arxiv.org/abs/2502.02464)
- [Time-R1: Temporal Reasoning with RL](https://arxiv.org/abs/2501.12345) (2025)
