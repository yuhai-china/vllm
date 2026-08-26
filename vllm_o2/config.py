# SPDX-License-Identifier: Apache-2.0
"""O2 model configuration: Qwen3.5 hybrid backbone + dual-expert MoE FFN.

`o2_text` is `qwen3_5_text` with the dense FFN replaced by `num_experts`
routed SwiGLU experts and a per-layer gate (no shared expert).
"""

from transformers.configuration_utils import PretrainedConfig

from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig


class O2TextConfig(Qwen3_5TextConfig):
    model_type = "o2_text"

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
    }

    def __init__(
        self,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=None,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = (
            moe_intermediate_size
            if moe_intermediate_size is not None
            else self.intermediate_size
        )
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef


class O2Config(Qwen3_5Config):
    model_type = "o2"
    sub_configs = {
        "vision_config": Qwen3_5VisionConfig,
        "text_config": O2TextConfig,
    }

    # NOTE: transformers v5's PreTrainedConfig.__init_subclass__ replaces the
    # inherited __init__ with an auto-generated dataclass __init__ unless the
    # subclass defines its own — which would skip the dict->sub-config
    # conversion in Qwen3_5Config.__init__. Keep an explicit __init__.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


__all__ = ["O2Config", "O2TextConfig"]
