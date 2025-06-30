#!/usr/bin/env python
"""
Quick demo of Ring Convolution Networks
Run this to see RCN in action!
"""

import numpy as np
from ring_convolution import RingConvolutionNetwork

print("=" * 60)
print("Ring Convolution Networks - Quick Demo")
print("=" * 60)

# Create a simple RCN
print("\n1. Creating RCN model...")
model = RingConvolutionNetwork(
    input_size=784,    # MNIST image size
    hidden_size=128,   
    output_size=10,    # 10 digits
    ring_depth=2,
    step_size=0.1
)
print("✓ Model created successfully!")

# Test on random "image"
print("\n2. Testing on random input...")
random_image = np.random.randn(784)
output = model(random_image)
predicted_class = np.argmax(output)

print(f"✓ Prediction: Class {predicted_class}")
print(f"✓ Confidence scores: {output[:3]}... (showing first 3)")

# Show ring weight properties
print("\n3. Ring weight properties:")
sample_weight = model.layer1.weights[0][0]
print(f"✓ Center value: {sample_weight.center:.4f}")
print(f"✓ Superposition: {sample_weight.compute_superposition():.4f}")
print(f"✓ Ring depth: {sample_weight.depth}")
print(f"✓ Step size: {sample_weight.step}")

print("\n" + "=" * 60)
print("Success! RCN is working properly.")
print("Run 'python experiments.py --compare-baseline' for full results")
print("=" * 60)
