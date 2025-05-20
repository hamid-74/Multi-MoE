# Multi-MoE: QoS-Efficient Serving of Multiple Mixture-of-Expert LLMs

Official repository for the paper "QoS-Efficient Serving of Multiple Mixture-of-Expert LLMs Using Partial Runtime Reconfiguration" accepted at ICML 2025.

## Overview

This repository provides the implementation of a novel serving system for efficiently deploying multiple fine-tuned Mixture-of-Experts (MoE) Large Language Models on a single GPU. The system introduces two key innovations:

1. **Similarity-based Expert Consolidation**: Reduces memory footprint by sharing similar experts across models.
2. **Runtime Partial Reconfiguration**: Dynamically replaces non-expert layers when processing requests from different models.

Our approach achieves competitive output quality while maintaining throughput comparable to serving a single model, with negligible increases in time-to-first-token (TTFT).

## Features

- Memory-efficient serving of multiple MoE LLMs on a single GPU
- Support for Mixtral-8x7B models (base and instruction-tuned variants)
- Automatic expert similarity detection and consolidation
- Dynamic layer-swapping capabilities for multi-model serving
- Evaluation on benchmarks including MT-Bench, MMLU, HellaSwag, and TruthfulQA

## Requirements

The code has been tested with the following packages:

```
python 3.9
torch 2.1.2
transformers 4.36.2
accelerate 0.26.1
huggingface-hub 0.24.6
datasets 2.21.0
tqdm 4.66.5
matplotlib 3.9.2
```

For a complete list of dependencies, see `packages_list.txt`.

## Usage

### Serving Multiple Models

The main implementation is in `src/MultiMoE.py`, which provides the functionality to serve multiple MoE models efficiently.

```python
from transformers import AutoTokenizer
from src.MultiMoE import MultiMoE

# Define models and configuration
model_ids = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
model_layout = {"non_expert": "mistralai/Mixtral-8x7B-v0.1"}

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# Create MultiMoE instance
multimoe = MultiMoE(args, model_ids, model_layout, tokenizer)

# Generate text
logits, exec_stats, outputs = multimoe.generate(
    text="Your prompt here", 
    output_token=100,
    temperature=0.7, 
    do_sample=True
)
```

### Evaluating on Benchmarks

The repository includes scripts for evaluating on several benchmarks:

- MT-Bench: `generate_mt_bench_responses.py`
- HellaSwag: `hellaswag_evaluator.py`
- MMLU: `mmlu_eval.py`
- TruthfulQA: `truthfulqa_evaluator.py`
- Perplexity evaluation: `ppl_evaluator.py`

### Quality of Service (QoS) Evaluation

To evaluate QoS metrics:
- Single GPU: `qos_evaluator_single.py`
- Multi-Instance GPU (MIG) configuration 1: `qos_evaluator_MIG1.py`
- Multi-Instance GPU (MIG) configuration 2: `qos_evaluator_MIG2.py`
- Plot QoS metrics: `qos_plotter.py`

## Results

The paper demonstrates that our approach achieves:
- 85% average reduction in turnaround time compared to NVIDIA's Multi-Instance GPU (MIG)
- Competitive output quality compared to individual model serving
- Scalability with multiple model variants

Result files can be found in the `results/` directory.

## Visualizations

- Expert distribution heat map: `expert_dist_heat_map.py` and `expert_dist_heat_map.png`
- Expert mapping visualization: `plot_expert_map.py` and `expert_map.png`
- Time-to-first-token plotter: `TTFT_plotter.py` and `ttft_turnaround.png`
- Scalability plotting: `plot_scalability.py` and `scalability.png`

## Citation

If you find this work useful for your research, please cite our paper:
```
@misc{imani2025qosefficientservingmultiplemixtureofexpert,
      title={QoS-Efficient Serving of Multiple Mixture-of-Expert LLMs Using Partial Runtime Reconfiguration}, 
      author={HamidReza Imani and Jiaxin Peng and Peiman Mohseni and Abdolah Amirany and Tarek El-Ghazawi},
      year={2025},
      eprint={2505.06481},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.06481}, 
}
```

## License

See the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [hamidreza@gwu.edu](mailto:hamidreza@gwu.edu).