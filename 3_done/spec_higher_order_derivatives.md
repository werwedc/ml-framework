# Spec: Higher-Order Derivatives

## Overview
Implement computation of higher-order derivatives (Jacobians and Hessians) to support meta-learning, optimization algorithms, and scientific computing applications.

## Files to Create
- `src/MLFramework/Autograd/Jacobian.cs`
- `src/MLFramework/Autograd/Hessian.cs`
- `src/MLFramework/Autograd/HigherOrderContext.cs`
- `tests/MLFramework.Tests/Autograd/HigherOrderTests.cs`

## API Design

### Class: Jacobian
```csharp
public static class Jacobian
{
    public static Tensor Compute(Func<Tensor, Tensor> f, Tensor x);
    public static Tensor Compute(Func<Tensor[], Tensor> f, Tensor[] inputs);
    public static Tensor[] ComputeVectorValued(Func<Tensor, Tensor[]> f, Tensor x);

    // Batch Jacobian for multiple inputs
    public static Tensor[] ComputeBatch(Func<Tensor, Tensor> f, Tensor[] inputs);
}
```

### Class: Hessian
```csharp
public static class Hessian
{
    public static Tensor Compute(Func<Tensor, double> f, Tensor x);
    public static Tensor ComputeDiagonal(Func<Tensor, double> f, Tensor x);
    public static Tensor ComputeVectorHessianProduct(Func<Tensor, double> f, Tensor x, Tensor v);
}
```

### Class: HigherOrderContext : IDisposable
```csharp
public class HigherOrderContext : IDisposable
{
    public bool CreateGraph { get; set; }
    public int MaxOrder { get; set; }

    public HigherOrderContext(bool createGraph = true, int maxOrder = 2);
    public void EnableGraphRetention();
    public void DisableGraphRetention();
    public void Dispose();
}
```

### Extension Methods
```csharp
public static class HigherOrderExtensions
{
    public static Tensor Jacobian(this Tensor tensor, Func<Tensor, Tensor> f);
    public static Tensor Hessian(this Tensor tensor, Func<Tensor, double> f);
    public static Tensor GradOfGrad(this Tensor tensor);
}
```

## Usage Examples

### Simple Jacobian
```csharp
// f(x) = x^2, J = 2x
var x = Tensor.FromArray(new double[] {1.0, 2.0, 3.0}, requiresGrad: true);
var f = new Func<Tensor, Tensor>(t => t.Pow(2));

var jacobian = Jacobian.Compute(f, x);
// Result: [2.0, 4.0, 6.0]
```

### Vector-Valued Jacobian
```csharp
// f(x) = [sin(x), cos(x)]
var x = Tensor.FromArray(new double[] {0.0, Math.PI/2}, requiresGrad: true);
var f = new Func<Tensor, Tensor[]>(t => new Tensor[] { t.Sin(), t.Cos() });

var jacobian = Jacobian.ComputeVectorValued(f, x);
// Result: [[cos(x), cos(x)], [-sin(x), -sin(x)]]
```

### Hessian Matrix
```csharp
// f(x) = x^4 + y^4
var x = Tensor.FromArray(new double[] {1.0, 2.0}, requiresGrad: true);
var f = new Func<Tensor, double>(t => t.Pow(4).Sum().ToScalar());

var hessian = Hessian.Compute(f, x);
// Result: [[12*x^2, 0], [0, 12*y^2]]
```

### Diagonal Hessian (memory efficient)
```csharp
var diagHessian = Hessian.ComputeDiagonal(f, x);
// Only returns diagonal elements: [12*x^2, 12*y^2]
```

### Gradient of Gradient (Meta-Learning)
```csharp
var theta = Tensor.Random(10, requiresGrad: true);
var x = Tensor.Random(5, 5, requiresGrad: true);

// Inner gradient
var y = model.Forward(theta, x);
var innerLoss = lossFn(y);
innerLoss.Backward(retainGraph: true);

// Outer gradient (gradient of gradient)
var gradTheta = theta.Grad.Clone().Detach().RequiresGrad();
var outerLoss = gradTheta.Sum();
outerLoss.Backward();

// Now theta.Grad contains gradient of gradient
```

## Requirements

### Core Functionality
1. **Jacobian Computation**
   - Compute Jacobian for scalar-valued functions
   - Compute Jacobian for vector-valued functions
   - Handle batch Jacobian computation
   - Support sparse Jacobian computation

2. **Hessian Computation**
   - Compute full Hessian matrix
   - Compute diagonal Hessian (memory efficient)
   - Compute Hessian-vector products
   - Handle large Hessians efficiently

3. **Graph Retention**
   - Retain computation graph for higher-order derivatives
   - Control graph retention with context
   - Dispose graph when no longer needed
   - Support arbitrary order derivatives

4. **Efficiency**
   - Use vector-Jacobian products (VJP) for efficiency
   - Use Hessian-vector products (HVP) for large Hessians
   - Cache intermediate results
   - Support numerical approximation fallback

## Implementation Notes

### Jacobian Computation
- For scalar f(x): J = ∇f(x) (gradient)
- For vector f(x): J[i,j] = ∂f_i/∂x_j
- Use reverse-mode AD (efficient for n << m)
- Use forward-mode AD for n >> m

### Hessian Computation
- H = ∇²f(x) (matrix of second derivatives)
- H[i,j] = ∂²f/∂x_i∂x_j
- Symmetric matrix (H = H^T)
- Use gradient-of-gradient approach
- For large Hessians, use diagonal approximation

### Graph Retention Strategy
- Default: clear graph after first backward
- For higher-order: set `retainGraph: true`
- Use HigherOrderContext for automatic management
- Clean up graphs explicitly to avoid memory leaks

### Numerical Stability
- Use central differences for approximation
- Handle near-zero denominators
- Add regularization for ill-conditioned Hessians
- Support analytical derivatives where available

## Testing Requirements

### Unit Tests
1. Compute Jacobian for f(x) = x^2 → verify J = 2x
2. Compute Jacobian for f(x) = sin(x) → verify J = cos(x)
3. Compute vector-valued Jacobian → verify shape
4. Compute Hessian for f(x,y) = x^2 + y^2 → verify H = 2I
5. Compute diagonal Hessian → verify matches full Hessian diagonal
6. Test gradient-of-gradient → verify correctness
7. Test batch Jacobian → verify correct batch handling
8. Test Hessian-vector product → verify HVP correctness

### Integration Tests
1. Compute Jacobian for neural network → verify shape
2. Compute Hessian for quadratic function → verify analytical match
3. Test meta-learning scenario (gradient of gradient)
4. Compare with numerical derivatives (finite difference)
5. Test higher-order derivatives for deep network
6. Benchmark Jacobian/Hessian computation time

## Dependencies
- Backward pass implementation (with graph retention)
- Tensor operations
- Computational graph infrastructure
- Linear algebra operations

## Success Criteria
- Accurate Jacobian/Hessian computation (within 1e-6)
- Efficient computation for large tensors
- Memory-efficient for sparse/diagonal cases
- Supports arbitrary order derivatives
- Clean API for common use cases
- Graph retention doesn't cause memory leaks
