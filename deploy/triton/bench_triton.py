#!/usr/bin/env python3
"""
Triton Inference Benchmark Script (Native Triton API)
Measures vision inference latency for nutrition label detection
"""
import requests
import time
import json
from statistics import mean
from PIL import Image
import io
import base64

# Load test image
print("Loading test image...")
img = Image.open("/models/test.jpg")
if max(img.size) > 1024:
    ratio = 1024 / max(img.size)
    img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)))

buffered = io.BytesIO()
img.save(buffered, format="JPEG")
img_base64 = base64.b64encode(buffered.getvalue()).decode()

print(f"Image loaded: {img.size[0]}x{img.size[1]}")
print("Starting benchmark...\n")

# Prepare vLLM-style messages for vision input
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            {"type": "text", "text": "Detect the nutrition table. Provide bounding box in [x_min, y_min, x_max, y_max] format."}
        ]
    }
]

# Triton inference request payload
payload = {
    "inputs": [
        {
            "name": "text_input",
            "shape": [1],
            "datatype": "BYTES",
            "data": [json.dumps({"messages": messages, "max_tokens": 200, "temperature": 0.01})]
        },
        {
            "name": "stream",
            "shape": [1],
            "datatype": "BOOL",
            "data": [False]
        }
    ]
}

# Benchmark
latencies = []
for i in range(5):
    start = time.time()
    try:
        response = requests.post(
            "http://localhost:8000/v2/models/qwen/generate",
            json=payload,
            timeout=120
        )

        latency = (time.time() - start) * 1000
        latencies.append(latency)

        if response.status_code == 200:
            result_data = response.json()
            # Extract text output from Triton response
            output_text = result_data['outputs'][0]['data'][0]
            print(f"{i+1}: {latency:.0f}ms - {output_text}")
        else:
            print(f"{i+1}: Failed - HTTP {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"{i+1}: Error - {e}")

# Save results
if latencies:
    results = {
        "mean_ms": round(mean(latencies)),
        "min_ms": round(min(latencies)),
        "max_ms": round(max(latencies)),
        "count": len(latencies)
    }
    print(f"\n{'='*60}")
    print(f"Mean latency: {results['mean_ms']}ms")
    print(f"Min latency:  {results['min_ms']}ms")
    print(f"Max latency:  {results['max_ms']}ms")
    print(f"{'='*60}")

    with open('/models/v1_latency.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /models/v1_latency.json")
else:
    print("No successful requests!")
