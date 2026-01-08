# Hessian Computation Tutorial

## Introduction

The Hessian matrix is the matrix of second-order partial derivatives of a scalar-valued function. It provides crucial information about the curvature of the function.

### Definition

Given a scalar function **f**: ℝⁿ → ℝ, the Hessian matrix H is an n×n symmetric matrix:

```
H = [∂²f/∂x₁²      ∂²f/∂x₁∂x₂    ...  ∂²f/∂x₁∂xₙ]
    [∂²f/∂x₂∂x₁   ∂²f/∂x₂²       ...  ∂²f/∂x₂∂xₙ]
    [   ...             ...               ...      ...   ]
    [∂²f/∂xₙ∂x₁   ∂²f/∂xₙ∂x₂    ...  ∂²f/∂xₙ²   ]
```

Key properties:
- **Symmetric**: H = Hᵀ (if f is twice continuously differentiable)
- **Positive Definite** at minima (for local minima)
- **Eigenvalues**: Curvature along principal directions

## When to Use Hessian

### 1. Optimization Algorithms
- **Newton's Method**: Use Hessian for faster convergence
- **Trust Region**: Model curvature locally
- **Line Search**: Use curvature for step size selection

### 2. Model Analysis
- **Sharpness**: Measure how flat minima are
- **Saddle Points**: Identify using eigenvalue signs
- **Landscape Analysis**: Understand loss surface geometry

### 3. Robustness
- **Adversarial Examples**: Find sensitive directions
- **Stability**: Analyze small perturbations
- **Generalization**: Flat minima generalize better

## Computing Full Hessian

### Basic Computation

```csharp
using MLFramework.Autograd;

// Define loss function
Func<Tensor, double> lossFunc = (Tensor parameters) =>
{
    var predictions = model.Forward(inputs);
    return ComputeLoss(predictions, targets);
};

// Compute full Hessian
Tensor hessian = Autograd.Hessian(
    loss: lossFunc,
    parameters: parameters
);

// Shape: (n, n) where n is total parameter count
Console.WriteLine($"Hessian shape: {hessian.Shape[0]} x {hessian.Shape[1]}");
```

### Example: Simple Quadratic Function

```csharp
// f(x, y) = x² + 2xy + y²
Func<Tensor[], double> quadratic = (Tensor[] params) =>
{
    float x = TensorAccessor.GetData(params[0])[0];
    float y = TensorAccessor.GetData(params[1])[0];
    return x * x + 2 * x * y + y * y;
};

var parameters = new Tensor[]
{
    Tensor.FromArray(new float[] {1.0f}),  // x
    Tensor.FromArray(new float[] {1.0f})   // y
};

Tensor hessian = Autograd.Hessian(
    p => quadratic(new[] {p[0], p[1]}),
    parameters[0]  // Compute w.r.t. x and y
);

// Expected Hessian:
// H = [[2, 2]
//      [2, 2]]
```

## Hessian-Vector Product (HVP)

The Hessian-Vector Product computes H*v without materializing the full Hessian matrix. This is crucial for large models.

### Why Use HVP?

1. **Memory Efficiency**: O(n) memory vs. O(n²) for full Hessian
2. **Computational Efficiency**: Similar cost to two gradient passes
3. **Sufficient for Many Applications**: Often only need H*v, not full H

### Computation

```csharp
// Define loss function
Func<Tensor, double> lossFunc = (Tensor parameters) =>
{
    return ComputeLoss(parameters);
};

// Vector to multiply with Hessian
Tensor vector = Tensor.FromArray(new float[] {1.0f, 0.5f, ...});

// Compute H*v
Tensor hvp = Autograd.HessianVectorProduct(
    loss: lossFunc,
    parameters: parameters,
    vector: vector
);
```

### Example: Power Iteration for Eigenvalues

```csharp
// Compute top eigenvalue using power iteration with HVP
Tensor ComputeTopEigenvalue(
    Func<Tensor, double> lossFunc,
    Tensor parameters,
    int iterations = 100)
{
    int n = parameters.Size;
    var rand = new Random(42);

    // Initialize random vector
    var v = new float[n];
    for (int i = 0; i < n; i++)
    {
        v[i] = (float)(rand.NextDouble() * 2 - 1);
    }
    var vTensor = new Tensor(v, new[] { n });

    // Power iteration
    for (int iter = 0; iter < iterations; iter++)
    {
        // Compute H*v using HVP
        Tensor hvp = Autograd.HessianVectorProduct(
            lossFunc,
            parameters,
            vTensor
        );

        var hvpData = TensorAccessor.GetData(hvp);

        // Normalize
        float norm = 0;
        for (int i = 0; i < n; i++)
        {
            norm += hvpData[i] * hvpData[i];
        }
        norm = (float)Math.Sqrt(norm);

        for (int i = 0; i < n; i++)
        {
            v[i] = hvpData[i] / norm;
        }

        // Update vTensor
        var vTensorData = TensorAccessor.GetData(vTensor);
        Array.Copy(v, vTensorData, n);
    }

    // Compute eigenvalue (Rayleigh quotient)
    // λ = vᵀ * H * v / vᵀ * v
    Tensor hvpFinal = Autograd.HessianVectorProduct(
        lossFunc,
        parameters,
        vTensor
    );

    var vData = TensorAccessor.GetData(vTensor);
    var hvpData = TensorAccessor.GetData(hvpFinal);

    float numerator = 0;
    float denominator = 0;
    for (int i = 0; i < n; i++)
    {
        numerator += vData[i] * hvpData[i];
        denominator += vData[i] * vData[i];
    }

    float eigenvalue = numerator / denominator;
    return Tensor.FromArray(new[] { eigenvalue });
}
```

## Diagonal Hessian

When you only need diagonal elements, compute them directly for efficiency.

### When to Use Diagonal Hessian

1. **Preconditioners**: Use diagonal Hessian as preconditioner
2. **Variance Estimation**: Estimate parameter uncertainty
3. **Quick Analysis**: Get rough curvature estimate

```csharp
// Compute diagonal Hessian
Tensor diagonalHessian = Autograd.Hessian(
    loss: lossFunc,
    parameters: parameters,
    options: new HessianOptions
    {
        ComputeEigenvalues = false,
        Sparse = false
    }
);

// Extract diagonal
var hessianData = TensorAccessor.GetData(diagonalHessian);
int n = diagonalHessian.Shape[0];
var diagonal = new float[n];
for (int i = 0; i < n; i++)
{
    diagonal[i] = hessianData[i * n + i];
}

Tensor diagTensor = new Tensor(diagonal, new[] { n });
```

## Eigenvalue Computation

### Using Power Iteration

Power iteration approximates the top eigenvalue efficiently:

```csharp
Tensor ComputeTopEigenvalue(
    Func<Tensor, double> lossFunc,
    Tensor parameters,
    int iterations = 100)
{
    // ... (see power iteration example above)
}
```

### Using Full Diagonalization

For small problems, compute all eigenvalues:

```csharp
// Compute Hessian with eigenvalues
var result = Autograd.Hessian(
    loss: lossFunc,
    parameters: parameters,
    options: new HessianOptions
    {
        ComputeEigenvalues = true,
        EigenvalueMethod = EigenvalueMethod.PowerIteration
    }
);

Tensor hessian = result.Hessian;
Tensor eigenvalues = result.Eigenvalues!;

// Analyze eigenvalues
var eigData = TensorAccessor.GetData(eigenvalues);
Console.WriteLine($"Top eigenvalue: {eigData[0]:F6}");
Console.WriteLine($"Bottom eigenvalue: {eigData[eigData.Length - 1]:F6}");
Console.WriteLine($"Condition number: {eigData[0] / eigData[eigData.Length - 1]:F2}");
```

### Eigenvalue Interpretation

| Eigenvalue | Interpretation |
|-------------|----------------|
| λ > 0 | Convex direction (valley) |
| λ < 0 | Concave direction (hill) |
| λ ≈ 0 | Flat direction |
| Large λ | Sharp curvature |
| Small λ | Gentle curvature |

### Example: Sharpness Analysis

```csharp
// Analyze sharpness of minima
Tensor eigenvalues = ComputeTopEigenvalue(lossFunc, parameters, 100);

float topEigenvalue = TensorAccessor.GetData(eigenvalues)[0];
float sharpness = topEigenvalue;

Console.WriteLine($"Sharpness (top eigenvalue): {sharpness:F6}");

// Rule of thumb:
// - Sharpness < 10: Very flat minima (good generalization)
// - 10 < Sharpness < 100: Moderately flat
// - Sharpness > 100: Sharp minima (poor generalization)
```

## Memory-Efficient Approximations

### Low-Rank Hessian

Approximate Hessian as low-rank matrix:

```csharp
// H ≈ U * diag(d) * Uᵀ
// where U is n×k and k << n

// This is used in methods like:
// - L-BFGS: Low-rank secant updates
// - K-FAC: Kronecker-factored approximate curvature
```

### Diagonal Plus Low-Rank

```csharp
// H ≈ diag(d) + U * diag(d_lr) * Uᵀ

// Combines simplicity of diagonal with accuracy of low-rank
```

## Practical Examples

### 1. Newton's Method Optimization

```csharp
// Newton's method using Hessian
void NewtonOptimization(
    Func<Tensor, double> lossFunc,
    Tensor parameters,
    int iterations = 100,
    float learningRate = 1.0f,
    float damping = 1e-4f)
{
    for (int iter = 0; iter < iterations; iter++)
    {
        // Compute gradient and Hessian
        Tensor grad = Autograd.Gradient(lossFunc, parameters);
        Tensor hessian = Autograd.Hessian(lossFunc, parameters);

        // Solve (H + λI) * Δ = -g
        Tensor delta = SolveNewtonStep(hessian, grad, damping);

        // Update parameters
        var paramData = TensorAccessor.GetData(parameters);
        var deltaData = TensorAccessor.GetData(delta);
        for (int i = 0; i < paramData.Length; i++)
        {
            paramData[i] += learningRate * deltaData[i];
        }

        Console.WriteLine($"Iter {iter}, Loss: {lossFunc(parameters):F6}");
    }
}
```

### 2. Sharpness-Aware Training

```csharp
// Find flat minima using Hessian
void SharpnessAwareTraining(
    Func<Tensor, double> lossFunc,
    Tensor parameters,
    int iterations = 100,
    float learningRate = 0.01f,
    float sharpnessRadius = 0.1f)
{
    for (int iter = 0; iter < iterations; iter++)
    {
        // Find direction of maximum curvature
        Tensor topEigen = ComputeTopEigenvalue(lossFunc, parameters, 100);

        // Perturb parameters in this direction
        var paramData = TensorAccessor.GetData(parameters);
        var eigData = TensorAccessor.GetData(topEigen);

        for (int i = 0; i < paramData.Length; i++)
        {
            paramData[i] += sharpnessRadius * eigData[i];
        }

        // Compute loss at perturbed parameters
        float perturbedLoss = lossFunc(parameters);

        // Update to minimize both losses
        Tensor grad = Autograd.Gradient(lossFunc, parameters);
        var gradData = TensorAccessor.GetData(grad);

        for (int i = 0; i < paramData.Length; i++)
        {
            paramData[i] -= learningRate * gradData[i];
        }

        Console.WriteLine($"Iter {iter}, Loss: {lossFunc(parameters):F6}");
    }
}
```

### 3. Adversarial Example Generation

```csharp
// Find sensitive input direction using Hessian
Tensor GenerateAdversarialExample(
    Func<Tensor, Tensor> model,
    Tensor input,
    Tensor trueLabel,
    float epsilon = 0.01f)
{
    // Compute loss
    Tensor prediction = model.Forward(input);
    Func<Tensor, double> lossFunc = (Tensor x) =>
    {
        Tensor pred = model.Forward(x);
        return ComputeLoss(pred, trueLabel);
    };

    // Compute Hessian w.r.t. input
    Tensor hessian = Autograd.Hessian(lossFunc, input);

    // Find top eigenvector
    Tensor topEigen = ComputeTopEigenvector(hessian, 100);

    // Perturb input in this direction
    var inputData = TensorAccessor.GetData(input);
    var eigData = TensorAccessor.GetData(topEigen);

    var perturbedData = new float[inputData.Length];
    for (int i = 0; i < inputData.Length; i++)
    {
        perturbedData[i] = inputData[i] + epsilon * eigData[i];
    }

    return new Tensor(perturbedData, input.Shape);
}
```

## Performance Tips

### 1. Use HVP When Possible

```csharp
// SLOW: Compute full Hessian
Tensor hessian = Autograd.Hessian(lossFunc, parameters);
Tensor hvp = MatrixVectorMultiply(hessian, vector);

// FAST: Compute HVP directly
Tensor hvp = Autograd.HessianVectorProduct(lossFunc, parameters, vector);
```

### 2. Use Conjugate Gradient for Newton Steps

```csharp
// SLOW: Explicitly invert Hessian
Tensor hessianInv = MatrixInverse(hessian);
Tensor delta = MatrixVectorMultiply(hessianInv, gradient);

// FAST: Solve H*x = g using CG
Tensor delta = ConjugateGradientSolver.Solve(hessian, gradient);
```

### 3. Leverage Sparsity

```csharp
// If Hessian is sparse, use sparse storage
Tensor sparseHessian = Autograd.Hessian(
    lossFunc,
    parameters,
    options: new HessianOptions { Sparse = true }
);
```

## Common Pitfalls

### 1. Numerical Instability

```csharp
// WRONG: Hessian may be ill-conditioned
Tensor delta = SolveLinearSystem(hessian, gradient);

// CORRECT: Add damping for stability
Tensor delta = SolveLinearSystem(
    MatrixAdd(hessian, MatrixIdentity(n) * damping),
    gradient
);
```

### 2. Memory Blow-up

```csharp
// WRONG: Computing full Hessian for large models
Tensor hugeHessian = Autograd.Hessian(lossFunc, millionParams);
// Memory: O(10¹²) elements!

// CORRECT: Use HVP or approximations
Tensor hvp = Autograd.HessianVectorProduct(lossFunc, parameters, vector);
// Memory: O(10⁶)
```

### 3. Forgetting Symmetry

```csharp
// WRONG: Computing Hessian incorrectly (not symmetric)
Tensor asymmetricHessian = ComputeHessianNaive();

// CORRECT: Use proper Hessian (should be symmetric)
Tensor hessian = Autograd.Hessian(lossFunc, parameters);
// Hessian should satisfy: H[i,j] = H[j,i]
```

## Numerical Verification

Always verify Hessian computations:

```csharp
// Verify Hessian using finite differences
Tensor ComputeNumericalHessian(
    Func<Tensor, double> lossFunc,
    Tensor parameters,
    double epsilon = 1e-6)
{
    int n = parameters.Size;
    var paramData = TensorAccessor.GetData(parameters);
    var hessianData = new float[n * n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float originalI = paramData[i];
            float originalJ = paramData[j];

            // f(x + εᵢ + εⱼ)
            paramData[i] = originalI + (float)epsilon;
            paramData[j] = originalJ + (float)epsilon;
            double f_pp = lossFunc(parameters);

            // f(x + εᵢ - εⱼ)
            paramData[i] = originalI + (float)epsilon;
            paramData[j] = originalJ - (float)epsilon;
            double f_pm = lossFunc(parameters);

            // f(x - εᵢ + εⱼ)
            paramData[i] = originalI - (float)epsilon;
            paramData[j] = originalJ + (float)epsilon;
            double f_mp = lossFunc(parameters);

            // f(x - εᵢ - εⱼ)
            paramData[i] = originalI - (float)epsilon;
            paramData[j] = originalJ - (float)epsilon;
            double f_mm = lossFunc(parameters);

            // Restore
            paramData[i] = originalI;
            paramData[j] = originalJ;

            // Second derivative
            hessianData[i * n + j] = (float)(
                (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon * epsilon)
            );
        }
    }

    return new Tensor(hessianData, new[] { n, n });
}

// Compare automatic and numerical Hessian
Tensor autoHessian = Autograd.Hessian(lossFunc, parameters);
Tensor numHessian = ComputeNumericalHessian(lossFunc, parameters);

var autoData = TensorAccessor.GetData(autoHessian);
var numData = TensorAccessor.GetData(numHessian);

float maxError = 0;
for (int i = 0; i < autoData.Length; i++)
{
    float error = Math.Abs(autoData[i] - numData[i]);
    maxError = Math.Max(maxError, error);
}

Console.WriteLine($"Max numerical error: {maxError:F8}");
```

## Summary

- **Hessian**: Matrix of second derivatives, provides curvature information
- **Full Hessian**: O(n²) memory, use for small problems
- **HVP**: O(n) memory, efficient for large problems
- **Diagonal Hessian**: Quick approximation for preconditioning
- **Eigenvalues**: Reveal sharpness and landscape structure
- **Choose Wisely**: Full Hessian vs. HVP based on problem size
- **Numerical Stability**: Add damping, verify with finite differences

## References

- Nocedal, J. & Wright, S. (2006). "Numerical Optimization"
- Martens, J. (2020). "New Insights and Perspectives on the Natural Gradient Method"
- Dauphin, Y. et al. (2014). "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization"
