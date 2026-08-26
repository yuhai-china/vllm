# vllm-o2

O2 model plugin for vLLM — a hybrid (linear/full attention) multimodal
backbone with a per-layer routed dual-expert MoE FFN.

## Install

```bash
pip install .
```

The plugin registers itself through vLLM's `vllm.general_plugins` entry
point; no other setup is needed.

## Serve

```bash
vllm serve /path/to/o2-checkpoint --served-model-name o2
```

Checkpoint layout:

```
model.language_model.layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight
model.language_model.layers.N.mlp.gate.weight
```
