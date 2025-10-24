import streamlit as st
import psutil
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- Helper to get resource usage ---
def get_resource_usage():
    process = psutil.Process(os.getpid())
    cpu = psutil.cpu_percent(interval=1)
    mem = process.memory_info().rss / (1024 * 1024)  # MB
    sys_mem = psutil.virtual_memory()
    return {
        "CPU (%)": cpu,
        "Process RAM (MB)": mem,
        "System RAM Used (GB)": sys_mem.used / 1e9,
        "System RAM (%)": sys_mem.percent
    }

# --- Dummy functions to simulate steps ---
def dummy_load_captions():
    time.sleep(1)
    with open('desc.txt', 'r') as f:
        lines = f.readlines()
    clean_desc = {}
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            key, caption = parts
            if key not in clean_desc:
                clean_desc[key] = []
            clean_desc[key].append(caption)
    return clean_desc

def dummy_clean_captions(clean_desc):
    time.sleep(1)
    return clean_desc

def dummy_import_model():
    time.sleep(1)
    return "Model Imported"

def dummy_feature_extraction():
    time.sleep(1)
    features_file = 'features.p'
    if os.path.exists(features_file):
        features = pickle.load(open(features_file, 'rb'))
        all_vecs = np.vstack(list(features.values()))
        return all_vecs
    else:
        return None

st.set_page_config(page_title="Resource Metrics & Graph", layout="wide")
st.title("CPU and Memory Utilization by Step")

steps = [
    ("Loading Captions", dummy_load_captions),
    ("Cleaning Captions", dummy_clean_captions),
    ("Importing Model", dummy_import_model),
    ("Feature Extraction", dummy_feature_extraction)
]

results = []
data_obj = None
cpu_vals = []
mem_vals = []
step_names = []

for step_name, func in steps:
    usage_before = get_resource_usage()
    if step_name == "Cleaning Captions":
        data_obj = func(data_obj)
    elif step_name == "Feature Extraction":
        data_obj = func()
    else:
        data_obj = func()
    usage_after = get_resource_usage()
    results.append({
        "Step": step_name,
        "CPU Before (%)": usage_before["CPU (%)"],
        "CPU After (%)": usage_after["CPU (%)"],
        "Process RAM Before (MB)": usage_before["Process RAM (MB)"],
        "Process RAM After (MB)": usage_after["Process RAM (MB)"],
        "System RAM Before (GB)": usage_before["System RAM Used (GB)"],
        "System RAM After (GB)": usage_after["System RAM Used (GB)"],
        "System RAM Before (%)": usage_before["System RAM (%)"],
        "System RAM After (%)": usage_after["System RAM (%)"]
    })
    cpu_vals.append(usage_after["CPU (%)"])
    mem_vals.append(usage_after["Process RAM (MB)"])
    step_names.append(step_name)

st.header("Resource Usage Table")
st.write("CPU and memory usage before and after each major function call.")
st.table(results)

st.header("CPU and Memory Utilization Graph")
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(step_names, cpu_vals, marker='o', color='green', label='CPU (%)')
ax1.set_ylabel('CPU Usage (%)', color='green')
ax1.set_xlabel('Step')
ax2 = ax1.twinx()
ax2.plot(step_names, mem_vals, marker='s', color='blue', label='RAM (MB)')
ax2.set_ylabel('Process RAM (MB)', color='blue')
plt.title('CPU and Memory Usage Across Steps')
fig.tight_layout()
st.pyplot(fig)
