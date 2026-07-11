import json
import os

STATS_FILE = "stats/ashkv_stats.json"

def main():
    if not os.path.exists(STATS_FILE):
        print(f"Stats file {STATS_FILE} not found. Did the server run with ASH-KV patches? Did it experience evictions?")
        return

    with open(STATS_FILE, "r") as f:
        try:
            stats = json.load(f)
        except json.JSONDecodeError:
            print(f"Stats file {STATS_FILE} is corrupt or empty.")
            return

    print("=== ASH-KV TELEMETRY PROOF ===")
    print(f"Blocks Intercepted:      {stats.get('blocks_intercepted', 0):,}")
    print(f"Total Tokens Compressed: {stats.get('tokens_compressed', 0):,}")
    print(f"Total Tokens Decompress: {stats.get('tokens_decompressed', 0):,}")
    
    bytes_saved = stats.get('bytes_saved', 0)
    mb_saved = bytes_saved / (1024 * 1024)
    gb_saved = mb_saved / 1024
    
    if gb_saved > 1:
        print(f"PCIe Bandwidth Saved:    {gb_saved:.2f} GB")
    else:
        print(f"PCIe Bandwidth Saved:    {mb_saved:.2f} MB")
    print("==============================")

if __name__ == "__main__":
    main()
