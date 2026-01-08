# Spec: Jacobian-Vector Product (JVP) Implementation

## Overview
Implement Jacobian-vector product (JVP) computation using forward-mode automatic differentiation. JVP complements VJP and is more efficient when the input dimension is smaller than the output dimension.

## Requirements

### Core Functionality
- Compute JVP: J * v where J is the Jacobian and v is a vector
- Support arbitrary input dimensionality
- Efficient computation using forward-mode differentiation
- Enable batch JVP computation for multiple vectors

### API Design
```csharp
// Basic JVP computation
Tensor jvp = Autograd.JacobianVectorProduct(function, parameters, vector);

// Batch JVP for multiple vectors
Tensor[] jvpBatch = Autograd.JacobianVectorProduct(function, parameters, vectorBatch);

// In-place computation with memory efficiency
Autograd.JacobianVectorProduct(function, parameters, vector, outputBuffer);
```

### Technical Details

#### Mathematical Foundation
- Given function f: R^n → R^m with parameters p
- Jacobian J is m×n matrix: J_ij = ∂f_i/∂p_j
- JVP computes: J * v where v ∈ R^n
- Result is a vector in R^m

#### Implementation Strategy
1. Implement forward-mode automatic differentiation (dual numbers)
2. Extend tensor operations to support dual number arithmetic
3. Implement JVP method using forward-mode computation
4. Leverage forward-mode for efficient JVP when n << m

#### Dual Number Arithmetic
- Represent tensors as dual numbers: (value, tangent)
- Define arithmetic operations on dual numbers:
  - Addition: (a, a') + (b, b') = (a+b, a'+b')
  - Multiplication: (a, a') * (b, b') = (a*b, a'*b + a*b')
  - Chain rule: f((a, a')) = (f(a), f'(a)*a')
- Extend all tensor operations to handle dual numbers

#### Forward-Mode Computation
- Initialize parameter tangents from input vector v
- Propagate tangents forward through computation graph
- Tangents at output nodes represent JVP
- Implement efficient tangent propagation for each operation

#### Optimization Techniques
- Skip zero tangents (sparsity exploitation in input vector)
- Parallelize tangent propagation for independent operations
- Implement tangent pooling for memory efficiency
- Cache tangent computation results for repeated JVPs

## Implementation Tasks

### Phase 1: Dual Number Infrastructure
1. Create DualTensor class wrapping (value, tangent) pairs
2. Extend basic arithmetic operations (+, -, *, /) for dual tensors
3. Implement advanced operations (matmul, conv, activation) for dual tensors
4. Add dual tensor unit tests

### Phase 2: Forward-Mode AD Engine
1. Implement forward-mode differentiation context
2. Add tangent propagation logic for computation graph
3. Implement JVP method using forward-mode engine
4. Add basic JVP unit tests

### Phase 3: Batch JVP Support
1. Implement batch vector multiplier handling
2. Add parallel tangent propagation for independent vectors
3. Optimize memory usage for batch operations
4. Add batch JVP unit tests

### Phase 4: Optimization Features
1. Implement sparsity exploitation (skip zero tangents)
2. Add tangent pooling and caching
3. Implement automatic JVP/VJP mode selection
4. Add performance profiling and benchmarks

## Testing Requirements

### Correctness Tests
- Test JVP against numerical differentiation for simple functions (linear, quadratic)
- Test JVP for multi-input functions (input dimension > 1)
- Test JVP for functions with complex computation graphs
- Test batch JVP consistency with individual JVP computations

### Edge Cases
- Test with zero vector multiplier
- Test with single-element vector multiplier
- Test with all-zero tangent scenarios
- Test with very large vector multipliers (> 1000 dimensions)

### Performance Tests
- Benchmark JVP vs VJP for different input/output dimensions
- Test memory usage for large computation graphs
- Benchmark batch JVP scaling with number of vectors
- Compare JVP performance for n << m vs n >> m scenarios

## Dependencies
- Extended gradient tape (spec_gradient_tape_extension.md)
- Tensor operations (basic arithmetic, linear algebra)
- Computation graph infrastructure
- Parallel computation framework (for batch operations)

## Success Criteria
- JVP matches numerical differentiation within 1e-6 tolerance for simple functions
- JVP computation time < VJP time when n << m (input dim < output dim)
- Support vector multipliers up to 10,000 dimensions
- Batch JVP scales linearly with number of vectors
- Memory overhead for dual tensors < 50% of standard tensors

## Notes for Coder
- Dual number arithmetic is the core of forward-mode AD - get this right
- Consider implementing operation-specific tangent propagation kernels (e.g., conv, matmul)
- Add comprehensive input validation (vector dimensions must match input)
- Document the forward-mode propagation algorithm
- Test thread safety for parallel batch JVP computation
- Consider implementing automatic mode selection: use JVP when n << m, VJP when m << n
