import pytest
import torch
from peft import LoraConfig, PeftModel, TaskType
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from training.train import model as model_mod
from training.train.config import smoke_config


def _tiny_distilbert() -> DistilBertForSequenceClassification:
    """A randomly-initialized 2-layer DistilBERT.

    Built from config — no network. Small enough that gradient checks run in
    milliseconds.
    """
    cfg = DistilBertConfig(
        vocab_size=200,
        max_position_embeddings=64,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        num_labels=3,
    )
    return DistilBertForSequenceClassification(cfg)


def test_default_lora_config_matches_supplement():
    cfg = model_mod.default_lora_config()
    assert isinstance(cfg, LoraConfig)
    assert cfg.task_type == TaskType.SEQ_CLS
    assert cfg.r == 8
    assert cfg.lora_alpha == 16
    assert set(cfg.target_modules) == {"q_lin", "v_lin"}
    assert "pre_classifier" in cfg.modules_to_save
    assert "classifier" in cfg.modules_to_save
    assert cfg.bias == "none"


def test_apply_lora_returns_a_peft_model():
    base = _tiny_distilbert()
    wrapped = model_mod.apply_lora(base, model_mod.default_lora_config())
    assert isinstance(wrapped, PeftModel)


def test_apply_lora_keeps_pre_classifier_and_classifier_trainable():
    base = _tiny_distilbert()
    wrapped = model_mod.apply_lora(base, model_mod.default_lora_config())

    trainable_names = {n for n, p in wrapped.named_parameters() if p.requires_grad}
    # The full pre_classifier and classifier weights must be trainable
    # (modules_to_save), not just LoRA adapters of them.
    assert any("pre_classifier" in n for n in trainable_names)
    assert any("classifier" in n for n in trainable_names)


def test_apply_lora_freezes_base_encoder_weights():
    base = _tiny_distilbert()
    wrapped = model_mod.apply_lora(base, model_mod.default_lora_config())

    frozen_names = {n for n, p in wrapped.named_parameters() if not p.requires_grad}
    # The base attention/FFN weights must be frozen — LoRA's whole point.
    assert any("attention.q_lin.base_layer" in n or "attention.q_lin.weight" in n
               for n in frozen_names)


def test_apply_lora_produces_three_class_logits():
    base = _tiny_distilbert()
    wrapped = model_mod.apply_lora(base, model_mod.default_lora_config())
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    out = wrapped(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (1, 3)


@pytest.mark.integration
def test_build_model_downloads_distilbert_and_wraps_it():
    cfg = smoke_config()
    wrapped = model_mod.build_model(cfg)
    assert isinstance(wrapped, PeftModel)
    # The real DistilBERT has 6 layers; the LoRA wrapper must see them.
    base_layers = [n for n, _ in wrapped.named_parameters() if "transformer.layer" in n]
    assert len(base_layers) > 0
