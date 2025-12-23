#!/usr/bin/env python3
"""
vLLM Vision Inference Benchmark Script
Measures inference latency for nutrition label detection on Qwen2-VL-7B
"""
import requests
import time
import json
from statistics import mean
from PIL import Image
import io
import base64

# Load and prepare test image
img = Image.open("/workspace/test.jpg")
if max(img.size) > 1024:
    ratio = 1024 / max(img.size)
    img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)))

buffered = io.BytesIO()
img.save(buffered, format="JPEG")
img_base64 = base64.b64encode(buffered.getvalue()).decode()

print(f"Image: {img.size[0]}x{img.size[1]}")
print("Starting benchmark...\n")

# Run benchmark (5 requests)
latencies = []
for i in range(5):
    start = time.time()
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "kulsoom-abdullah/qwen2-7b-nutrition-labels-detection",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    {"type": "text", "text": "Detect the nutrition table. Provide bounding box in [x_min, y_min, x_max, y_max] format."}
                ]
            }],
            "max_tokens": 200,
            "temperature": 0.01
        },
        timeout=120
    )

    latency = (time.time() - start) * 1000

    if response.status_code == 200:
        latencies.append(latency)
        result = response.json()['choices'][0]['message']['content']
        print(f"{i+1}: {latency:.0f}ms - {result}")
    else:
        print(f"{i+1}: Failed - HTTP {response.status_code}")

# Save results
if latencies:
    results = {
        "mean_ms": round(mean(latencies)),
        "min_ms": round(min(latencies)),
        "max_ms": round(max(latencies)),
        "count": len(latencies)
    }
    print(f"\nMean: {results['mean_ms']}ms, Min: {results['min_ms']}ms, Max: {results['max_ms']}ms")

    with open("/workspace/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to /workspace/results.json")
else:
    print("No successful requests!")
