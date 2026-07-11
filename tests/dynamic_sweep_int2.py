import torch
import torch.nn.functional as F
from ashkv.codecs.int2_dithered import DitheredINT2Codec

def inject_outliers(tensor: torch.Tensor, outlier_ratio: float, magnitude: float):
    if outlier_ratio == 0.0:
        return tensor
        
    num_elements = tensor.numel()
    num_outliers = int(num_elements * outlier_ratio)
    
    indices = torch.randperm(num_elements, device=tensor.device)[:num_outliers]
    signs = torch.randint(0, 2, (num_outliers,), device=tensor.device).float() * 2 - 1
    
    flat_tensor = tensor.view(-1)
    flat_tensor[indices] += signs * magnitude * flat_tensor.std().item()
    
    return flat_tensor.view_as(tensor)

def evaluate_config(k_adv, v_adv, q, group_size, sigma):
    head_dim = k_adv.shape[-1]
    codec = DitheredINT2Codec(hidden_dim=head_dim, group_size=group_size, sigma=sigma)
    
    bf16_bytes = k_adv.numel() * 2 + v_adv.numel() * 2
    
    k_bytes = codec.encode(k_adv.contiguous().view(torch.int16).cpu().numpy().tobytes())
    v_bytes = codec.encode(v_adv.contiguous().view(torch.int16).cpu().numpy().tobytes())
    
    compressed_bytes = len(k_bytes) + len(v_bytes)
    
    k_restored_bytes = codec.decode(k_bytes)
    v_restored_bytes = codec.decode(v_bytes)
    
    k_restored = torch.frombuffer(bytearray(k_restored_bytes), dtype=torch.bfloat16).cuda().view_as(k_adv)
    v_restored = torch.frombuffer(bytearray(v_restored_bytes), dtype=torch.bfloat16).cuda().view_as(v_adv)
    
    attn_base = F.scaled_dot_product_attention(
        q.unsqueeze(0).transpose(1, 2), 
        k_adv.unsqueeze(0).transpose(1, 2), 
        v_adv.unsqueeze(0).transpose(1, 2)
    )
    
    attn_int2 = F.scaled_dot_product_attention(
        q.unsqueeze(0).transpose(1, 2), 
        k_restored.unsqueeze(0).transpose(1, 2), 
        v_restored.unsqueeze(0).transpose(1, 2)
    )
    
    cos_sim = F.cosine_similarity(attn_base.float().view(-1), attn_int2.float().view(-1), dim=0).item()
    comp_ratio = bf16_bytes / compressed_bytes
    
    return cos_sim, comp_ratio

def run_sweep():
    print("[SWEEP] Booting Intelligent Dynamic Parameter Sweep for Dithered INT2...\n")
    
    seq_len = 1024
    head_num = 8
    head_dim = 128 # Using 128 to test group sizes up to 64
    
    environments = {
        "Smooth Normal": {"ratio": 0.0, "mag": 0.0},
        "Mild Adversarial": {"ratio": 0.01, "mag": 20.0},
        "Extreme Heavy-Tail": {"ratio": 0.03, "mag": 100.0}
    }
    
    group_sizes = [16, 32, 64]
    sigmas = [2.0, 2.5, 3.0, 3.5, 4.0]
    
    torch.manual_seed(14)
    
    for env_name, params in environments.items():
        print(f"==================================================")
        print(f" ENVIRONMENT: {env_name.upper()}")
        print(f" Outlier Ratio: {params['ratio']*100}%, Magnitude: {params['mag']}x")
        print(f"==================================================")
        
        k_normal = torch.randn(seq_len, head_num, head_dim, dtype=torch.bfloat16, device="cuda") * 0.02
        v_normal = torch.randn(seq_len, head_num, head_dim, dtype=torch.bfloat16, device="cuda") * 0.02
        q = torch.randn(1, head_num, head_dim, dtype=torch.bfloat16, device="cuda") * 0.02
        
        k_adv = inject_outliers(k_normal.clone(), params["ratio"], params["mag"])
        v_adv = inject_outliers(v_normal.clone(), params["ratio"], params["mag"])
        
        best_config = None
        best_fitness = 0.0
        
        print(f"{'Group':<6} | {'Sigma':<6} | {'Cos Sim':<10} | {'Ratio':<8} | {'Fitness'}")
        print("-" * 50)
        
        for g in group_sizes:
            for s in sigmas:
                cos_sim, comp_ratio = evaluate_config(k_adv, v_adv, q, group_size=g, sigma=s)
                
                # Fitness Function: Maximize compression ratio ONLY IF fidelity > 0.99
                fitness = comp_ratio if cos_sim >= 0.99 else 0.0
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_config = (g, s, cos_sim, comp_ratio)
                
                fit_str = f"{fitness:.2f}" if fitness > 0 else "FAIL"
                print(f"{g:<6} | {s:<6.1f} | {cos_sim:<10.6f} | {comp_ratio:<7.2f}x | {fit_str}")
        
        if best_config:
            bg, bs, bc, br = best_config
            print(f"\n=> OPTIMAL CONFIG for {env_name}: Group={bg}, Sigma={bs}")
            print(f"   Yields {br:.2f}x compression at {bc:.6f} fidelity.\n")
        else:
            print(f"\n=> OPTIMAL CONFIG for {env_name}: NONE (All failed >0.99 threshold)\n")

if __name__ == "__main__":
    run_sweep()
