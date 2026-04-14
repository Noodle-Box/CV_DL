import torch
# This lab calculates gradients for a simple algorithm.
# Implements the use of back propagation to compute gradients given 5 random inputs and a given L (final output)

def compute_gradients(w0, x0, w1, x1, w2):

    # Foward pass
    s0 = w0 * x0
    s1 = w1 * x1
    s2 = s0 + s1
    s3 = s2 + w2
    L = torch.sigmoid(s3)

    # Backward pass

    #Base
    grad_L = 1.0

    #Sigmoid
    grad_s3 = grad_L * L * (1 - L)

    #Add gate 2
    grad_w2 = grad_s3
    grad_s2 = grad_s3

    #Add gate 1
    grad_s0 = grad_s2
    grad_s1 = grad_s2

    #Multiply gate 2
    grad_w1 = grad_s1 * x1
    grad_x1 = grad_s1 * w1

    #Multiply gate 1
    grad_w0 = grad_s0 * x0
    grad_x0 = grad_s0 * w0

    return grad_w0, grad_x0, grad_w1, grad_x1, grad_w2

# ============== Calculate Gradients with Random Inputs ==============

print("\n" + "="*60)
print("GRADIENT CALCULATION WITH RANDOM INPUTS")
print("="*60)

# Define 5 random inputs
w0 = 2.0
x0 = -1.0
w1 = -3.0
x1 = -2.0
w2 = -3.0

print(f"\nRandom Input Values:")
print(f"  w0 = {w0:.2f}")
print(f"  x0 = {x0:.2f}")
print(f"  w1 = {w1:.2f}")
print(f"  x1 = {x1:.2f}")
print(f"  w2 = {w2:.2f}")

# Convert inputs to tensors for compute_gradients function
w0_tensor = torch.tensor(w0, dtype=torch.float32)
x0_tensor = torch.tensor(x0, dtype=torch.float32)
w1_tensor = torch.tensor(w1, dtype=torch.float32)
x1_tensor = torch.tensor(x1, dtype=torch.float32)
w2_tensor = torch.tensor(w2, dtype=torch.float32)

# Calculate gradients using the compute_gradients function
grad_w0, grad_x0, grad_w1, grad_x1, grad_w2 = compute_gradients(w0_tensor, x0_tensor, w1_tensor, x1_tensor, w2_tensor)

print(f"\nComputed Gradients:")
print(f"  ∂L/∂w0 = {grad_w0:.2f}")
print(f"  ∂L/∂x0 = {grad_x0:.2f}")
print(f"  ∂L/∂w1 = {grad_w1:.2f}")
print(f"  ∂L/∂x1 = {grad_x1:.2f}")
print(f"  ∂L/∂w2 = {grad_w2:.2f}")

print("\n" + "="*60)
