"""
Tucker Decomposition for Transformer Weight Compression
Source: Linear Algebra / Tensor Analysis (physics applications)

Compresses MLP weights in transformer layers using Tucker decomposition.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorly.decomposition import tucker
import json
import os
from pathlib import Path


def tucker_decompose_tensor(tensor, ranks):
    """
    Apply Tucker decomposition to a 4D tensor (e.g., conv or reshaped linear weights).
    
    Args:
        tensor: Input tensor of shape (out_features, in_features)
        ranks: Target ranks for each mode (r1, r2)
    
    Returns:
        core: Core tensor
        factors: List of factor matrices
    """
    # Reshape 2D weight to higher dimensional if needed for Tucker
    # For linear layers: (out_features, in_features) -> treat as 2D tensor
    if len(tensor.shape) == 2:
        # Use truncated SVD as special case of Tucker for 2D
        U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
        r = min(ranks[0], len(S))
        U_r = U[:, :r]
        S_r = torch.diag(S[:r])
        Vh_r = Vh[:r, :]
        
        # Reconstructed: U_r @ S_r @ Vh_r
        core = S_r
        factors = [U_r, Vh_r.T]
        return core, factors
    else:
        # For true multi-dimensional tensors
        tensor_np = tensor.cpu().numpy()
        core, factors = tucker(tensor_np, ranks=ranks)
        return torch.from_numpy(core), [torch.from_numpy(f) for f in factors]


def compress_linear_layer(layer, compression_ratio=4.0):
    """
    Compress a linear layer using Tucker-inspired truncated SVD.
    
    Args:
        layer: nn.Linear layer
        compression_ratio: Target compression (e.g., 4.0 = 4x smaller)
    
    Returns:
        Compressed representation dict
    """
    weight = layer.weight.data
    in_features, out_features = weight.shape[1], weight.shape[0]
    
    # Calculate target rank for desired compression
    # Original params: in_features * out_features
    # Compressed: r * (in_features + out_features)
    # Solve: r = (in_features * out_features) / (compression_ratio * (in_features + out_features))
    original_params = in_features * out_features
    target_params = original_params / compression_ratio
    
    # r * (in_features + out_features) ≈ target_params
    # But need r <= min(in_features, out_features)
    r = int(target_params / (in_features + out_features))
    r = max(1, min(r, min(in_features, out_features) - 1))
    
    # Apply SVD
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    
    # Truncate
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    
    compressed = {
        'U': U_r,
        'S': S_r,
        'Vh': Vh_r,
        'bias': layer.bias.data if layer.bias is not None else None,
        'in_features': in_features,
        'out_features': out_features,
        'rank': r
    }
    
    # Verify reconstruction
    reconstructed = (U_r * S_r.unsqueeze(0)) @ Vh_r
    
    return compressed, reconstructed


def compress_mlp_layers(model, compression_ratios={'c_fc': 4.0, 'c_proj': 4.0}):
    """
    Compress MLP layers in GPT-style models.
    
    For distilgpt2/GPT2:
    - transformer.h[i].mlp.c_fc: projection up (768 -> 3072)
    - transformer.h[i].mlp.c_proj: projection down (3072 -> 768)
    
    Returns:
    compressed_state: Dict of compressed layers
    original_size_mb: Original model size
    compressed_size_mb: Compressed model size
    """
    compressed_state = {}
    original_params = 0
    compressed_params = 0
    
    print(f"Searching for MLP layers...")
    
    for name, module in model.named_modules():
        # GPT2 models use mlp.c_fc and mlp.c_proj
        if 'mlp.c_fc' in name or 'mlp.c_proj' in name:
            if isinstance(module, nn.Linear):
                layer_type = 'c_fc' if 'c_fc' in name else 'c_proj'
                ratio = compression_ratios.get(layer_type, 4.0)
                
                print(f"  Found {name}: {module.weight.shape}")
                
                compressed, reconstructed = compress_linear_layer(module, ratio)
                compressed_state[name] = compressed
                
                # Parameter counts
                orig_p = module.weight.numel()
                comp_p = compressed['U'].numel() + compressed['S'].numel() + compressed['Vh'].numel()
                
                original_params += orig_p
                compressed_params += comp_p
                
                # Replace weight with reconstructed for testing
                module.weight.data = reconstructed
    
    print(f"\nCompressed {len(compressed_state)} MLP layers")
    
    # Calculate sizes (4 bytes per float32)
    original_size_mb = (original_params * 4) / (1024 * 1024)
    compressed_size_mb = (compressed_params * 4) / (1024 * 1024)
    
    print(f"Original params: {original_params:,}, Compressed: {compressed_params:,}")
    
    return compressed_state, original_size_mb, compressed_size_mb


def evaluate_perplexity(model, tokenizer, texts=None):
    """
    Evaluate model perplexity on sample texts.
    """
    if texts is None:
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models are becoming more efficient daily.",
            "The capital of France is Paris.",
            "Water boils at 100 degrees Celsius at sea level.",
        ]
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                model = model.cuda()
            
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs['input_ids'].numel()
            total_tokens += inputs['input_ids'].numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def main():
    print("=" * 60)
    print("EXPERIMENT 01: Tucker Decomposition Compression")
    print("=" * 60)
    
    # Configuration
    MODEL_NAME = "distilgpt2"
    COMPRESSION_RATIOS = {'c_fc': 4.0, 'c_proj': 4.0}
    OUTPUT_DIR = Path("results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print(f"\n[1/5] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Baseline evaluation
    print("\n[2/5] Evaluating baseline (uncompressed)...")
    baseline_ppl, baseline_loss = evaluate_perplexity(model, tokenizer)
    print(f"Baseline perplexity: {baseline_ppl:.2f}")
    
    # Compress MLP layers
    print("\n[3/5] Applying Tucker decomposition to MLP layers...")
    compressed_state, orig_size, comp_size = compress_mlp_layers(model, COMPRESSION_RATIOS)
    
    print(f"Original MLP size: {orig_size:.2f} MB")
    print(f"Compressed MLP size: {comp_size:.2f} MB")
    if comp_size > 0:
        print(f"Compression ratio: {orig_size/comp_size:.2f}x")
    else:
        print(f"WARNING: Compressed size is 0, cannot calculate ratio")
    
    # Compressed evaluation
    print("\n[4/5] Evaluating compressed model...")
    compressed_ppl, compressed_loss = evaluate_perplexity(model, tokenizer)
    print(f"Compressed perplexity: {compressed_ppl:.2f}")
    
    # Calculate metrics
    ppl_increase = ((compressed_ppl - baseline_ppl) / baseline_ppl) * 100
    
    results = {
        'experiment': '01-tucker-decomposition',
        'model': MODEL_NAME,
        'compression_ratios': COMPRESSION_RATIOS,
        'baseline': {
            'perplexity': round(baseline_ppl, 4),
            'loss': round(baseline_loss, 6)
        },
        'compressed': {
            'perplexity': round(compressed_ppl, 4),
            'loss': round(compressed_loss, 6),
            'mlp_size_mb': round(comp_size, 4),
            'original_mlp_size_mb': round(orig_size, 4)
        },
        'metrics': {
            'compression_ratio': round(orig_size / comp_size, 2),
            'perplexity_increase_pct': round(ppl_increase, 2),
            'size_reduction_mb': round(orig_size - comp_size, 2)
        },
        'details': {
            'num_layers_compressed': len(compressed_state),
            'sample_layer_ranks': {k: v['rank'] for k, v in list(compressed_state.items())[:2]}
        }
    }
    
    # Save results
    print("\n[5/5] Saving results...")
    output_file = OUTPUT_DIR / '01-tucker-baseline.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Compression ratio: {results['metrics']['compression_ratio']}x")
    print(f"Perplexity increase: {results['metrics']['perplexity_increase_pct']:.2f}%")
    print(f"Size saved: {results['metrics']['size_reduction_mb']:.2f} MB")
    print(f"\nStatus: {'SUCCESS' if ppl_increase < 20 else 'REVIEW NEEDED'}")
    
    return results


if __name__ == "__main__":
    results = main()
