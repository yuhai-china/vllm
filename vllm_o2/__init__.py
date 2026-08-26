# vLLM plugin registering the O2 architecture (model_type "o2" / "o2_text").
# Loaded through the "vllm.general_plugins" entry point; must be idempotent
# because vLLM loads general plugins in multiple processes.

_registered = False


def register():
    global _registered
    if _registered:
        return

    try:
        # registers the O2 tokenizer/processor/config classes with
        # transformers so renamed checkpoint metadata keeps resolving
        import o2_model  # noqa: F401
    except ImportError:
        pass

    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        Qwen3_5ForConditionalGenerationConfig,
    )
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from .config import O2Config

    _CONFIG_REGISTRY["o2"] = O2Config
    ModelRegistry.register_model(
        "O2ForConditionalGeneration", "vllm_o2.model:O2ForConditionalGeneration"
    )
    MODELS_CONFIG_MAP.setdefault(
        "O2ForConditionalGeneration", Qwen3_5ForConditionalGenerationConfig
    )
    _registered = True


__all__ = ["register"]
