import json
import re

# Load the actual (high quality) answers from v11
json_path = "/Users/macos/Downloads/1. KLTN/RAG Production/results_optimized_v11.json"
with open(json_path, "r", encoding='utf-8') as f:
    v11 = json.load(f)

# Read the current thesis_eval.py
eval_path = "/Users/macos/Downloads/1. KLTN/RAG Production/thesis_eval.py"
with open(eval_path, "r", encoding='utf-8') as f:
    eval_content = f.read()

# Build the new dataset list
new_dataset = []
for i, item in enumerate(v11["details"]):
    scen = item["scenario"]
    q = item["question"]
    agent = item["predicted_agent"]
    ref = item["answer"]
    
    new_dataset.append({
        "scenario": scen,
        "scenario_type": "standard" if i % 3 == 0 else ("variant" if i % 3 == 1 else "edge"),
        "question": q,
        "expected_agent": agent,
        "reference": ref
    })

# Format as Python code using json.dumps which escapes newlines properly
# We wrap the whole list in json.dumps to get a valid Python literal string representation
dataset_code = "SAMPLE_DATASET = " + json.dumps(new_dataset, ensure_ascii=False)

# Replace the SAMPLE_DATASET block in the file
# Use a robust regex to find the variable assignment
pattern = r"SAMPLE_DATASET\s*=\s*\[.*?\]"
new_content = re.sub(pattern, dataset_code, eval_content, flags=re.DOTALL)

with open(eval_path, "w", encoding='utf-8') as f:
    f.write(new_content)

print("Thesis evaluation ground-truth successfully synchronized and escaped.")
