from __future__ import annotations

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from training.train.config import TrainConfig

_NUM_LABELS = 3  # weak / partial / strong


def default_lora_config() -> LoraConfig:
    """The Phase 3 LoRA config.

    `modules_to_save=["pre_classifier", "classifier"]` is load-bearing — both
    classification heads are randomly-initialized and must train as full
    layers, not as low-rank adapters. See the design supplement §3.1.
    """
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_lin", "v_lin"],
        modules_to_save=["pre_classifier", "classifier"],
        bias="none",
    )


def apply_lora(base_model: PreTrainedModel, lora_config: LoraConfig) -> PeftModel:
    """Wrap a sequence-classification model with LoRA."""
    return get_peft_model(base_model, lora_config)


def load_base_model(model_name: str, *, num_labels: int = _NUM_LABELS) -> PreTrainedModel:
    """Download (or load from cache) a sequence-classification model.

    Hits the network on first call. Tested as integration only.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )


def build_model(config: TrainConfig) -> PeftModel:
    """One-shot: load the base model and wrap it with the default LoRA config."""
    base = load_base_model(config.model_name)
    return apply_lora(base, default_lora_config())
