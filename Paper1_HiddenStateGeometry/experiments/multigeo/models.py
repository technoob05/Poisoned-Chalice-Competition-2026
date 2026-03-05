"""Model loading and GPU memory management."""
import gc
import torch


def load_model(model_name: str, dtype_str: str = "bfloat16"):
    """
    Load a causal LM with hidden states + attentions enabled.

    Returns: (model, tokenizer, n_layers)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n  Loading model: {model_name}")
    dtype = getattr(torch, dtype_str, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
        attn_implementation="eager",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  ✓ {model_name}: {n_layers}L, {n_params:.1f}B params, dtype={dtype}")
    return model, tokenizer, n_layers


def free_model(*objects):
    """Free GPU memory after model use."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"  ♻ GPU memory freed. Current: {allocated:.1f} GB")
