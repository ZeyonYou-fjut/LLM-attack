# SeedHijack: Code and Experimental Results

This repository contains the source code and experimental results for LLM attack.


## Directory Structure

```
├── code/
│   ├── run_experiments.py      # Main experiment script (Attack B, QRNG defense, performance)
│   ├── exp4_7b_attack_c.py     # Attack C on 7B-scale aligned models
│   ├── backdoor_sampler.py     # Backdoor sampler implementation (attack module)
│   ├── attack_eval.py          # Attack evaluation utilities
│   ├── qrng_defense.py         # QRNG defense module
│   ├── draw_figures.py         # Figure generation (Figure 2: heatmap, Figure 3: defense panels)
│   └── draw_figure1.py         # Figure 1 generation (attack overview)
├── results/
│   ├── exp1_attack_b.json      # Attack B results: 540 trials, 9 configs, 99.6% injection rate
│   ├── exp2_qrng_defense.json  # QRNG defense: 100% attack neutralization
│   ├── exp3_performance.json   # Overhead: +0.6% median latency, +7.7 MB memory
│   ├── exp4_attack_c.json      # Attack C on 1.5B aligned models (100% success)
│   ├── exp4_7b_attack_c.json   # Attack C on 7B aligned models (100% success)
│   └── exp4_7b_progress.log    # Experiment progress log
├── config.json                   # Experiment configuration and hyperparameters
└── README.md
```

## Requirements

- Python 3.9+
- PyTorch 2.1+
- Transformers 4.36+
- NVIDIA GPU with CUDA 11.8+
- QRNG600 PCIe hardware (for defense experiments; pre-buffered samples used in evaluation)

## Experiments

### Experiment 1: SeedHijack Attack (Attack B)
- **Script**: `code/run_experiments.py` (Experiment 1)
- **Model**: GPT-2 (124M)
- **Setup**: 60 trials × 9 sampling configurations (temperature ∈ {0.7, 1.0, 1.5} × top_p ∈ {0.9, 0.95, 1.0})
- **Result**: 99.6% exact token injection rate (538/540)

### Experiment 2: QRNG Defense
- **Script**: `code/run_experiments.py` (Experiment 2)
- **Model**: GPT-2 (124M)
- **Result**: Attack injection rate drops to 0% under QRNG-defended sampling

### Experiment 3: Performance Overhead
- **Script**: `code/run_experiments.py` (Experiment 3)
- **Result**: +0.6% median latency, +7.7 MB memory overhead

### Experiment 4: Cross-Model Attack (Attack C)
- **Script**: `code/run_experiments.py` (Experiment 4) + `code/exp4_7b_attack_c.py`
- **Models**: Qwen2-1.5B-Instruct (RLHF), DeepSeek-R1-Distill-1.5B (distillation), Qwen2-7B-Instruct, DeepSeek-R1-Distill-Qwen-7B
- **Result**: 100% injection rate on all aligned models

## Figures

To regenerate figures from results:
```bash
python code/draw_figures.py
python code/draw_figure1.py
```

## Hardware

- GPU: NVIDIA RTX 3090 (24 GB VRAM)
- QRNG: QRNG600 PCIe quantum random number generator card

## License

This code is provided for academic research purposes only.
