import torch
import torch.nn.functional as F
import struct
import sys
from ashkv.codecs.int2_dithered import DitheredINT2Codec

def inject_heavy_tail_outliers(tensor: torch.Tensor, outlier_ratio: float = 0.01, magnitude: float = 50.0):
    """Injects massive outliers into random indices of the tensor."""
    num_elements = tensor.numel()
    num_outliers = int(num_elements * outlier_ratio)
    
    # Generate random indices
    indices = torch.randperm(num_elements, device=tensor.device)[:num_outliers]
    
    # Randomly make them positive or negative spikes
    signs = torch.randint(0, 2, (num_outliers,), device=tensor.device).float() * 2 - 1
    
    flat_tensor = tensor.view(-1)
    # Multiply the existing variance by the magnitude
    flat_tensor[indices] += signs * magnitude * flat_tensor.std().item()
    
    return flat_tensor.view_as(tensor)

def run_stress_test():
    print("[STRESS TEST] Booting Adversarial INT2 Validation...")
    
    seq_len = 1024
    head_num = 8
    head_dim = 64
    embed_dim = head_num * head_dim
    
    codec = DitheredINT2Codec(hidden_dim=head_dim)
    
    # 1. Generate normal distribution (like standard activations)
    k_normal = torch.randn(seq_len, head_num, head_dim, dtype=torch.bfloat16, device="cuda") * 0.02
    v_normal = torch.randn(seq_len, head_num, head_dim, dtype=torch.bfloat16, device="cuda") * 0.02
    
    # 2. Inject massive 50x outliers (heavy-tailed)
    k_adv = inject_heavy_tail_outliers(k_normal.clone(), outlier_ratio=0.01, magnitude=50.0)
    v_adv = inject_heavy_tail_outliers(v_normal.clone(), outlier_ratio=0.01, magnitude=50.0)
    
    bf16_bytes = k_adv.numel() * 2 + v_adv.numel() * 2
    
    # 3. Encode
    k_bytes = codec.encode(k_adv.contiguous().view(torch.int16).cpu().numpy().tobytes())
    v_bytes = codec.encode(v_adv.contiguous().view(torch.int16).cpu().numpy().tobytes())
    
    compressed_bytes = len(k_bytes) + len(v_bytes)
    
    # 4. Decode
    k_restored_bytes = codec.decode(k_bytes)
    v_restored_bytes = codec.decode(v_bytes)
    
    k_restored = torch.frombuffer(bytearray(k_restored_bytes), dtype=torch.bfloat16).cuda().view_as(k_adv)
    v_restored = torch.frombuffer(bytearray(v_restored_bytes), dtype=torch.bfloat16).cuda().view_as(v_adv)
    
    # 5. Measure Fidelity (Simulate Attention)
    q = torch.randn(1, head_num, head_dim, dtype=torch.bfloat16, device="cuda") * 0.02
    
    # Baseline Attention (Infinite VRAM)
    attn_base = F.scaled_dot_product_attention(
        q.transpose(1, 2), 
        k_adv.unsqueeze(0).transpose(1, 2), 
        v_adv.unsqueeze(0).transpose(1, 2)
    )
    
    # Compressed Attention (Dithered INT2)
    attn_int2 = F.scaled_dot_product_attention(
        q.transpose(1, 2), 
        k_restored.unsqueeze(0).transpose(1, 2), 
        v_restored.unsqueeze(0).transpose(1, 2)
    )
    
    cos_sim = F.cosine_similarity(attn_base.float().view(-1), attn_int2.float().view(-1), dim=0).item()
    comp_ratio = bf16_bytes / compressed_bytes
    
    # 6. Verify Outliers were actually isolated
    # Extract num_outliers from the header (first 8 bytes: num_tokens, num_outliers)
    k_header = struct.unpack("<II", k_bytes[:8])
    v_header = struct.unpack("<II", v_bytes[:8])
    
    k_outliers_isolated = k_header[1]
    v_outliers_isolated = v_header[1]
    
    print("\n" + "="*50)
    print("      INT2 ADVERSARIAL STRESS TEST REPORT")
    print("="*50)
    print(f"Cosine Similarity:      {cos_sim:.6f} (Target > 0.99)")
    print(f"Compression Ratio:      {comp_ratio:.2f}x (Target > 4.0x)")
    print(f"K Outliers Isolated:    {k_outliers_isolated}")
    print(f"V Outliers Isolated:    {v_outliers_isolated}")
    print(f"Original BF16 size:     {bf16_bytes} bytes")
    print(f"Compressed INT2 size:   {compressed_bytes} bytes")
    print("="*50)
    
    assert cos_sim > 0.99, f"Validation Failed: Cosine similarity dropped to {cos_sim:.6f}"
    assert comp_ratio > 4.0, f"Validation Failed: Compression ratio dropped to {comp_ratio:.2f}x"
    assert k_outliers_isolated > 0 and v_outliers_isolated > 0, "Validation Failed: Outlier isolation mechanism did not trigger"
    
    print("\n[PASS] 3-Sigma Outlier Isolation successfully protected 2-bit scale against 50x magnitude spikes.")

if __name__ == "__main__":
    run_stress_test()
