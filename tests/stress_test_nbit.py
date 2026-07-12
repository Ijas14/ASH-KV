import torch
import torch.nn.functional as F
import struct
from ashkv.codecs.nbit_dithered import DitheredNBitCodec

def inject_heavy_tail_outliers(tensor: torch.Tensor, outlier_ratio: float = 0.01, magnitude: float = 50.0):
    num_elements = tensor.numel()
    num_outliers = int(num_elements * outlier_ratio)
    
    indices = torch.randperm(num_elements, device=tensor.device)[:num_outliers]
    signs = torch.randint(0, 2, (num_outliers,), device=tensor.device).float() * 2 - 1
    
    flat_tensor = tensor.view(-1)
    flat_tensor[indices] += signs * magnitude * flat_tensor.std().item()
    
    return flat_tensor.view_as(tensor)

def evaluate_nbit_codec(k_adv, v_adv, q, bits):
    head_dim = k_adv.shape[-1]
    codec = DitheredNBitCodec(hidden_dim=head_dim, bits=bits)
    
    bf16_bytes = k_adv.numel() * 2 + v_adv.numel() * 2
    
    k_bytes = codec.encode(k_adv.contiguous().view(torch.int16).cpu().numpy().tobytes())
    v_bytes = codec.encode(v_adv.contiguous().view(torch.int16).cpu().numpy().tobytes())
    
    compressed_bytes = len(k_bytes) + len(v_bytes)
    
    k_restored_bytes = codec.decode(k_bytes)
    v_restored_bytes = codec.decode(v_bytes)
    
    k_restored = torch.frombuffer(bytearray(k_restored_bytes), dtype=torch.bfloat16).cuda().view_as(k_adv)
    v_restored = torch.frombuffer(bytearray(v_restored_bytes), dtype=torch.bfloat16).cuda().view_as(v_adv)
    
    attn_base = F.scaled_dot_product_attention(
        q.transpose(0, 1), 
        k_adv.transpose(0, 1), 
        v_adv.transpose(0, 1)
    )
    
    attn_comp = F.scaled_dot_product_attention(
        q.transpose(0, 1), 
        k_restored.transpose(0, 1), 
        v_restored.transpose(0, 1)
    )
    
    cos_sim = F.cosine_similarity(attn_base.float().view(-1), attn_comp.float().view(-1), dim=0).item()
    comp_ratio = bf16_bytes / compressed_bytes
    
    return cos_sim, comp_ratio

def run_stress_test():
    print("[N-BIT STRESS TEST] Booting Adversarial Validation for N-Bit Codec...")
    
    seq_len = 1024
    head_num = 8
    head_dim = 64
    
    torch.manual_seed(42)
    
    k_normal = torch.randn(seq_len, head_num, head_dim, dtype=torch.bfloat16, device="cuda") * 0.02
    v_normal = torch.randn(seq_len, head_num, head_dim, dtype=torch.bfloat16, device="cuda") * 0.02
    
    # Inject massive 50x outliers
    k_adv = inject_heavy_tail_outliers(k_normal.clone(), outlier_ratio=0.01, magnitude=50.0)
    v_adv = inject_heavy_tail_outliers(v_normal.clone(), outlier_ratio=0.01, magnitude=50.0)
    
    q = torch.randn(seq_len, head_num, head_dim, dtype=torch.bfloat16, device="cuda") * 0.02
    
    results = {}
    for bits in [2, 4, 8]:
        sim, ratio = evaluate_nbit_codec(k_adv, v_adv, q, bits=bits)
        results[f"INT{bits}"] = {"sim": sim, "ratio": ratio}
    
    print("\n" + "="*60)
    print("      N-BIT ADVERSARIAL STRESS TEST REPORT")
    print("="*60)
    print(f"{'Codec':<10} | {'Cosine Similarity':<20} | {'Compression Ratio'}")
    print("-" * 60)
    
    for name, metrics in results.items():
        print(f"{name:<10} | {metrics['sim']:<20.6f} | {metrics['ratio']:<7.2f}x")
    print("="*60)
    
    assert results["INT2"]["sim"] > 0.98, "Validation Failed: INT2 dropped below 0.98"
    assert results["INT4"]["sim"] > 0.999, "Validation Failed: INT4 dropped below 0.999"
    assert results["INT8"]["sim"] > 0.9999, "Validation Failed: INT8 dropped below 0.9999"
    
    print("\n[PASS] Generalized N-Bit Codec proves exponential fidelity scaling with bit-width.")

if __name__ == "__main__":
    run_stress_test()
