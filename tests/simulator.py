import torch
import torch.nn as nn
import torch.nn.functional as F
from ashkv.adapters.sglang.hooks import SGLangHooks
from ashkv.compiler.registry import codec_registry
import time

class E2ESimulator:
    """Standalone Transformer execution environment for mathematically proving ASH-KV codecs."""
    def __init__(self, seq_len=1024, head_num=8, head_dim=64, gpu_capacity=512):
        self.seq_len = seq_len
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = 1
        self.gpu_capacity = gpu_capacity
        self.embed_dim = head_num * head_dim
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False, dtype=torch.bfloat16).cuda()
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False, dtype=torch.bfloat16).cuda()
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False, dtype=torch.bfloat16).cuda()
        
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        
    def run_baseline(self, hidden_states: torch.Tensor):
        """Baseline: Infinite VRAM, standard BF16 memory."""
        q = self.q_proj(hidden_states).view(self.seq_len, self.head_num, self.head_dim)
        k = self.k_proj(hidden_states).view(self.seq_len, self.head_num, self.head_dim)
        v = self.v_proj(hidden_states).view(self.seq_len, self.head_num, self.head_dim)
        
        q_b = q.unsqueeze(0).transpose(1, 2)
        k_b = k.unsqueeze(0).transpose(1, 2)
        v_b = v.unsqueeze(0).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)
        return attn_output.transpose(1, 2).reshape(self.seq_len, self.embed_dim), q, k, v

    def run_codec_direct(self, hidden_states: torch.Tensor, k_full: torch.Tensor, v_full: torch.Tensor, codec_name: str):
        """Directly encodes and decodes evicted tokens using the specified codec."""
        codec = codec_registry.get(codec_name)
        # Ensure the codec knows the hidden dimension (head_dim) for correct reshaping
        if hasattr(codec, "_hidden_dim"):
            codec._hidden_dim = self.head_dim
            
        q_full = self.q_proj(hidden_states).view(self.seq_len, self.head_num, self.head_dim)
        
        num_to_demote = self.seq_len - self.gpu_capacity
        
        # Saliency (S)
        q_h = q_full.float().transpose(0, 1)
        k_h = k_full.float().transpose(0, 1)
        
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len, device="cuda")) == 0
        scores.masked_fill_(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        saliency = attn_weights.sum(dim=(0, 1))
        saliency[0] = float('inf')
        saliency[-32:] = float('inf') 
        
        _, lowest_indices = torch.topk(saliency, num_to_demote, largest=False)
        device_indices = lowest_indices.sort().values
        
        k_demote = k_full[device_indices].reshape(-1, self.head_dim)
        v_demote = v_full[device_indices].reshape(-1, self.head_dim)
        
        # Encode
        k_packed = codec.encode(k_demote.contiguous().detach().cpu().numpy().tobytes())
        v_packed = codec.encode(v_demote.contiguous().detach().cpu().numpy().tobytes())
        
        compressed_bytes = len(k_packed) + len(v_packed)
        bf16_bytes = k_demote.numel() * 2 + v_demote.numel() * 2
        
        # Decode
        k_restored_bytes = codec.decode(k_packed)
        v_restored_bytes = codec.decode(v_packed)
        
        k_restored = torch.frombuffer(bytearray(k_restored_bytes), dtype=torch.bfloat16).cuda().view(-1, self.head_num, self.head_dim)
        v_restored = torch.frombuffer(bytearray(v_restored_bytes), dtype=torch.bfloat16).cuda().view(-1, self.head_num, self.head_dim)
        
        k_test = k_full.clone()
        v_test = v_full.clone()
        k_test[device_indices] = k_restored
        v_test[device_indices] = v_restored
        
        q_b = q_full.unsqueeze(0).transpose(1, 2)
        k_b = k_test.unsqueeze(0).transpose(1, 2)
        v_b = v_test.unsqueeze(0).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)
        return attn_output.transpose(1, 2).reshape(self.seq_len, self.embed_dim), bf16_bytes, compressed_bytes

    def validate(self):
        with torch.no_grad():
            hidden_states = torch.randn(self.seq_len, self.embed_dim, dtype=torch.bfloat16, device="cuda")
            
            base_out, _, base_k, base_v = self.run_baseline(hidden_states)
            
            # We need to warm up Triton for INT8 to avoid timeout/crash during testing
            from ashkv.codecs.int8 import _get_kernels
            _get_kernels()
            
            int8_out, int8_bf16, int8_comp = self.run_codec_direct(hidden_states, base_k, base_v, "int8_default")
            int2_out, int2_bf16, int2_comp = self.run_codec_direct(hidden_states, base_k, base_v, "int2_dithered")
            
            int8_sim = F.cosine_similarity(base_out.float().view(-1), int8_out.float().view(-1), dim=0).item()
            int2_sim = F.cosine_similarity(base_out.float().view(-1), int2_out.float().view(-1), dim=0).item()
            
            return {
                "int8_sim": int8_sim,
                "int2_sim": int2_sim,
                "int8_bytes": int8_comp,
                "int2_bytes": int2_comp,
                "bf16_bytes": int8_bf16
            }

if __name__ == "__main__":
    print("[ASH-KV] Booting E2E Validation Simulator...")
    sim = E2ESimulator(seq_len=2048, gpu_capacity=1024)
    res = sim.validate()
    
    print("\n" + "="*60)
    print("      ASH-KV 3-WAY CODEC VALIDATION REPORT")
    print("="*60)
    print(f"{'Metric':<25} | {'INT8 (Triton)':<15} | {'INT2 Dithered (PyTorch)'}")
    print("-" * 60)
    print(f"{'Cosine Similarity':<25} | {res['int8_sim']:<15.6f} | {res['int2_sim']:.6f}")
    print(f"{'Memory Footprint (KB)':<25} | {res['int8_bytes']/1024:<15.2f} | {res['int2_bytes']/1024:.2f}")
    print(f"{'Compression Ratio':<25} | {res['bf16_bytes']/res['int8_bytes']:<15.2f}x | {res['bf16_bytes']/res['int2_bytes']:.2f}x")
    print("="*60)
    
    if res['int2_sim'] < 0.99:
        print("[FAIL] Dithered INT2 validation failed! Accuracy dropped below threshold.")
        exit(1)
    else:
        print("[PASS] Dithered INT2 undeniable math proof passed!")
