# Jacobian Computation Tutorial

## Introduction

The Jacobian matrix is a fundamental concept in multivariate calculus that generalizes the gradient to vector-valued functions.

### Definition

Given a function **f**: ℝⁿ → ℝᵐ, the Jacobian matrix J is an m×n matrix of first-order partial derivatives:

```
J = [∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ]
    [∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ]
    [  ...        ...         ...      ...  ]
    [∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₙ]
```

For scalar-valued functions (m=1), the Jacobian reduces to the gradient:
```
∇f = [∂f/∂x₁  ∂f/∂x₂  ...  ∂f/∂xₙ]
```

## When to Use Jacobian vs. Gradient

### Use Gradient (Scalar Output)
- Loss functions (e.g., MSE, Cross-Entropy)
- Scalar metrics
- Optimization objectives

```csharp
// Loss: ℝⁿ → ℝ (scalar)
Tensor loss = ComputeLoss(predictions, targets);
Tensor grad = Autograd.Gradient(p => ComputeLoss(p, targets), parameters);
```

### Use Jacobian (Vector Output)
- Neural network outputs
- Multiple outputs
- Vector transformations

```csharp
// Network: ℝⁿ → ℝᵐ (vector)
Tensor output = model.Forward(input);
Tensor jacobian = Autograd.Jacobian(p => model.Forward(p), input);
```

## Jacobian-Vector Product (JVP)

The Jacobian-Vector Product computes J*v without materializing the full Jacobian matrix. This is much more efficient when v is small.

### Definition
```
JVP(J, v) = J * v
```

### When to Use JVP
- Forward-mode automatic differentiation
- Computing directional derivatives
- When the output dimension m is small

```csharp
// Compute J*v
Tensor vector = new Tensor(new float[] {1.0f, 0.0f, ...});  // Vector to multiply
Tensor jvp = Autograd.JacobianVectorProduct(
    f: x => model.Forward(x),
    x: input,
    vector: vector
);
```

### Example: Sensitivity Analysis

```csharp
// Analyze how output changes with respect to input direction
Tensor input = Tensor.FromArray(new float[] {1.0f, 2.0f, 3.0f});
Tensor direction = Tensor.FromArray(new float[] {0.707f, 0.707f, 0.0f});  // 45° in x-y plane

// Compute directional derivative
Tensor jvp = Autograd.JacobianVectorProduct(x => model.Forward(x), input, direction);

Console.WriteLine($"Directional derivative: {TensorAccessor.GetData(jvp)[0]:F6}");
```

## Vector-Jacobian Product (VJP)

The Vector-Jacobian Product computes vᵀ*J without materializing the full Jacobian. This is the standard backpropagation operation.

### Definition
```
VJP(J, v) = vᵀ * J
```

### When to Use VJP
- Reverse-mode automatic differentiation (backpropagation)
- When the input dimension n is small
- Computing gradients for optimization

```csharp
// Compute vᵀ*J (gradient)
Tensor vector = new Tensor(new float[] {1.0f, 0.0f});  // Upstream gradient
Tensor vjp = Autograd.VectorJacobianProduct(
    f: x => model.Forward(x),
    x: input,
    vector: vector
);
```

### Example: Gradient Computation

```csharp
// Compute gradient (VJP with v = [1, 0, ..., 0] for first output)
Tensor input = Tensor.FromArray(new float[] {1.0f, 2.0f});
Tensor upstreamGrad = Tensor.FromArray(new float[] {1.0f});  // ∂L/∂output

Tensor grad = Autograd.VectorJacobianProduct(
    f: x => model.Forward(x),
    x: input,
    vector: upstreamGrad
);

Console.WriteLine($"Gradient: [{string.Join(", ", TensorAccessor.GetData(grad))}]");
```

## Practical Use Cases

### 1. Neural Network Layer Analysis

```csharp
// Analyze sensitivity of each output to input
Tensor input = Tensor.FromArray(new float[] {1.0f, 2.0f, 3.0f});
Tensor output = model.Forward(input);

// Compute full Jacobian
Tensor jacobian = Autograd.Jacobian(x => model.Forward(x), input);

// Each row j of Jacobian shows sensitivity of output j to all inputs
for (int j = 0; j < output.Size; j++)
{
    float sensitivityNorm = 0;
    for (int i = 0; i < input.Size; i++)
    {
        float element = TensorAccessor.GetData(jacobian)[j * input.Size + i];
        sensitivityNorm += element * element;
    }
    Console.WriteLine($"Output {j} sensitivity: {Math.Sqrt(sensitivityNorm):F4}");
}
```

### 2. Input Attribution

```csharp
// Which input features contribute most to output?
Tensor input = Tensor.FromArray(new float[] {1.0f, 2.0f, 3.0f});
Tensor output = model.Forward(input);

// Compute gradient of output w.r.t. input (VJP)
Tensor vjp = Autograd.VectorJacobianProduct(
    f: x => model.Forward(x),
    x: input,
    vector: Tensor.Ones(new[] {output.Size})  // Sum over outputs
);

// Large absolute gradient = important feature
var gradData = TensorAccessor.GetData(vjp);
for (int i = 0; i < gradData.Length; i++)
{
    Console.WriteLine($"Feature {i} importance: {Math.Abs(gradData[i]):F4}");
}
```

### 3. Batch Processing

```csharp
// Compute Jacobians for a batch of inputs
var inputs = new Tensor[]
{
    Tensor.FromArray(new float[] {1.0f, 2.0f}),
    Tensor.FromArray(new float[] {2.0f, 1.0f}),
    Tensor.FromArray(new float[] {0.5f, 0.5f})
};

var jacobians = new List<Tensor>();
foreach (var input in inputs)
{
    Tensor jacobian = Autograd.Jacobian(x => model.Forward(x), input);
    jacobians.Add(jacobian);
}

// Analyze Jacobian statistics across batch
```

## Performance Considerations

### Memory vs. Computation Trade-off

| Method | Memory | Computation | When to Use |
|--------|---------|-------------|-------------|
| Full Jacobian | O(nm) | O(nm) | Small n, m |
| JVP | O(m) | O(n + m) | Small m |
| VJP | O(n) | O(n + m) | Small n |
| HVP | O(n) | O(n + m) | Vector needed |

### Choosing the Right Method

1. **Full Jacobian**: When you need the entire matrix (e.g., for visualization or analysis)
2. **JVP**: When you need directional derivatives (forward mode)
3. **VJP**: When you need gradients (reverse mode - backpropagation)

### Memory Efficiency

```csharp
// BAD: Computing full Jacobian for large dimensions
Tensor hugeJacobian = Autograd.Jacobian(x => model.Forward(x), input);  // O(nm) memory

// GOOD: Computing just the product you need
Tensor jvp = Autograd.JacobianVectorProduct(x => model.Forward(x), input, vector);  // O(m) memory
```

## Common Pitfalls

### 1. Wrong Dimensions

```csharp
// WRONG: Thinking Jacobian shape is (n, m)
// The Jacobian is (m, n) where f: ℝⁿ → ℝᵐ

// CORRECT: Check dimensions
int n = input.Size;   // Input dimension
int m = output.Size;  // Output dimension
Tensor jacobian = Autograd.Jacobian(x => model.Forward(x), input);
Console.WriteLine($"Jacobian shape: {jacobian.Shape[0]} x {jacobian.Shape[1]}");  // Should be m x n
```

### 2. Confusing JVP and VJP

```csharp
// WRONG: Using JVP when you need gradient
Tensor jvp = Autograd.JacobianVectorProduct(x => model.Forward(x), input, upstreamGrad);

// CORRECT: Using VJP for gradient (reverse mode)
Tensor vjp = Autograd.VectorJacobianProduct(x => model.Forward(x), input, upstreamGrad);
```

### 3. Forgetting Chain Rule

When composing functions:
```
If y = f(x) and z = g(y)
Then J_z = J_g * J_f
```

The Autograd framework handles this automatically, but be aware of the complexity.

## Numerical Verification

Always verify your Jacobian computations numerically:

```csharp
Tensor jacobian = Autograd.Jacobian(x => model.Forward(x), input);
Tensor numerical = ComputeNumericalJacobian(model, input);

var jacData = TensorAccessor.GetData(jacobian);
var numData = TensorAccessor.GetData(numerical);

float maxError = 0;
for (int i = 0; i < jacData.Length; i++)
{
    float error = Math.Abs(jacData[i] - numData[i]);
    maxError = Math.Max(maxError, error);
}

Console.WriteLine($"Max numerical error: {maxError:F8}");
```

## Advanced Topics

### Higher-Order Jacobians

```csharp
// Compute Jacobian of Jacobian (third-order tensor)
Tensor thirdOrder = Autograd.Jacobian(
    x => Autograd.Jacobian(y => model.Forward(y), x),
    input
);
```

### Batch Jacobians

```csharp
// Compute Jacobian for batch efficiently
// Use batching to parallelize computation
```

## Summary

- **Gradient**: For scalar functions (ℝⁿ → ℝ)
- **Jacobian**: For vector functions (ℝⁿ → ℝᵐ)
- **JVP**: Compute J*v efficiently (forward mode)
- **VJP**: Compute vᵀ*J efficiently (reverse mode)
- **Choose wisely**: Full Jacobian vs. products based on dimensions
- **Verify**: Always check numerically when implementing custom operations

## References

- Griewank, A. & Walther, A. (2008). "Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation"
- Baydin, A. G. et al. (2018). "Automatic Differentiation in Machine Learning: a Survey"
