import pandas as pd
import json
import glob
import matplotlib.pyplot as plt

data = []
for file in glob.glob("results_*.json"):
    with open(file) as f:
        result = json.load(f)
        result["vm_name"] = file.split("_")[1].split(".")[0]
        data.append(result)

df = pd.DataFrame(data)
df = df[[
    "vm_name", "execution_time_sec", "cpu_usage_percent", "gpu_usage_percent",
    "memory_consumption_MB", "model_accuracy", "inference_latency_ms"
]]

print("\n📈 Performance Comparison Table:\n")
print(df)

df.to_csv("vm_comparison_report.csv", index=False)
print("\n✅ Report saved as vm_comparison_report.csv")

# Plot each metric
df.plot(x="vm_name", kind="bar", subplots=True, layout=(3,2), figsize=(12,10),
        title="VM Performance Comparison Metrics")
plt.tight_layout()
plt.show()
