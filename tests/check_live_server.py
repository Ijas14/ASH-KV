import os
import sys
import time

LOG_FILE = "stats/ashkv_patch.log"

def main():
    print("=== ASH-KV Live Server Diagnostic ===")
    print("Checking if the Ray GPU worker successfully patched itself...\n")
    
    if not os.path.exists(LOG_FILE):
        print(f"❌ Error: {LOG_FILE} does not exist.")
        print("   The server either hasn't started, or sitecustomize.py didn't run.")
        sys.exit(1)
        
    with open(LOG_FILE, "r") as f:
        logs = f.read()
        
    print("--- RAW WORKER LOGS ---")
    print(logs.strip())
    print("-----------------------\n")
    
    if "Patches APPLIED successfully" in logs:
        print("✅ SUCCESS: The SGLang GPU worker is officially running ASH-KV Triton compression!")
        print("   You are cleared to run the benchmark.")
    elif "Failed to auto-patch" in logs:
        print("❌ FAILED: The worker tried to patch but hit an error.")
    else:
        print("⏳ WAITING: The worker started the background thread but hasn't applied the patches yet.")
        print("   SGLang might still be downloading or initializing the model weights.")

if __name__ == "__main__":
    main()
