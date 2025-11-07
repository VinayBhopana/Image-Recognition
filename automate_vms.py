from fabric import Connection
import json

# ---------- 5 VMs ------------
vms = [
    {"name": "vm1_tiny", "host": "10.0.0.11", "user": "ubuntu", "key": "key.pem"},
    {"name": "vm2_small", "host": "10.0.0.12", "user": "ubuntu", "key": "key.pem"},
    {"name": "vm3_medium", "host": "10.0.0.13", "user": "ubuntu", "key": "key.pem"},
    {"name": "vm4_gpu", "host": "10.0.0.14", "user": "ubuntu", "key": "key.pem"},
    {"name": "vm5_highend", "host": "10.0.0.15", "user": "ubuntu", "key": "key.pem"},
]
# ------------------------------

for vm in vms:
    print(f"\n🚀 Running benchmark on {vm['name']} ({vm['host']})")
    c = Connection(
        host=vm["host"],
        user=vm["user"],
        connect_kwargs={"key_filename": vm["key"]}
    )

    # Upload project files
    c.put("image_caption_model.py")
    c.put("benchmark_runner.py")
    c.put("tokenizer.p")
    c.put("desc.txt")
    # (You can upload dataset paths if small; better mount them)

    # Install dependencies
    c.run("pip install tensorflow keras psutil matplotlib pillow tqdm numpy")

    # Run benchmark
    c.run("python3 benchmark_runner.py")

    # Download result
    c.get("results.json", f"results_{vm['name']}.json")

print("\n✅ All 5 VM benchmarks completed.")
