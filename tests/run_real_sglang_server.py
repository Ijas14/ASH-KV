import sys
import argparse
from sglang.launch_server import launch_server
from ashkv.adapters.sglang.hooks import SGLangHooks
from ashkv.adapters.sglang.patcher import apply_hicache_patches

def main():
    # Parse the model argument
    parser = argparse.ArgumentParser(description="Start SGLang Server with ASH-KV Patches")
    parser.add_argument("--model", type=str, required=True, help="Path to model (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--port", type=int, default=30000, help="Port to run the server on")
    
    # We use parse_known_args in case the user passes standard sglang args
    args, unknown = parser.parse_known_args()

    print("--- INITIALIZING ASH-KV ---")
    hooks = SGLangHooks(codec_name="int8_default")
    
    # Apply patches BEFORE SGLang initializes
    print("Applying HiCache patches to SGLang HostKVCache...")
    apply_hicache_patches(hooks)
    print("Patches applied successfully.")
    
    print(f"--- STARTING SGLANG HTTP SERVER ---")
    print(f"Model: {args.model}")
    print(f"Port: {args.port}")
    
    # Construct sys.argv for the launch_server function
    sys.argv = [
        "launch_server",
        "--model-path", args.model,
        "--port", str(args.port),
        "--mem-fraction-static", "0.3",  # Force small cache for eviction testing
        "--hicache-write-policy", "write_through",  # Trigger for ASH-KV
        "--log-level", "info",
    ] + unknown
    
    # Launch the actual SGLang server!
    launch_server()

if __name__ == "__main__":
    main()
