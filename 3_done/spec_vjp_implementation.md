# Spec: Vector-Jacobian Product (VJP) Implementation

## Overview
Implement vector-Jacobian product (VJP) computation using reverse-mode automatic differentiation. VJP is a fundamental operation for computing full Jacobians efficiently and is the backbone of reverse-mode AD.

## Requirements

### Core Functionality
- Compute VJP: v^T * J where J is the Jacobian and v is a vector
- Support arbitrary output dimensionality
- Efficient computation leveraging existing backward pass infrastructure
- Enable batch VJP computation for multiple vectors

### API Design
```csharp
// Basic VJP computation
Tensor vjp = Autograd.VectorJacobianProduct(function, parameters, vector);

// Batch VJP for multiple vectors
Tensor[] vjpBatch = Autograd.VectorJacobianProduct(function, parameters, vectorBatch);

// In-place computation with memory efficiency
Autograd.VectorJacobianProduct(function, parameters, vector, outputBuffer);
```

### Technical Details

#### Mathematical Foundation
- Given function f: R^n → R^m with parameters p
- Jacobian J is m×n matrix: J_ij = ∂f_i/∂p_j
- VJP computes: v^T * J where v ∈ R^m
- Result is a vector in R^n

#### Implementation Strategy
1. Extend reverse-mode backward pass to accept arbitrary vector multipliers
2. Modify gradient accumulation to support weighted gradients
3. Leverage existing gradient computation graph for efficiency
4. Implement early termination for sparse vector multipliers

#### Computation Graph Traversal
- Use depth-first or breadth-first traversal for gradient propagation
- Implement gradient chain rule: dL/dp = ∑_i (∂f_i/∂p) * v_i
- Support intermediate gradient caching for repeated VJP computations
- Add gradient checkpointing for memory efficiency

#### Optimization Techniques
- Skip zero entries in vector multiplier (sparsity exploitation)
- Parallelize gradient computation for independent graph branches
- Implement automatic vector multiplier batching for efficiency
- Cache intermediate gradients for repeated VJPs

## Implementation Tasks

### Phase 1: Core VJP Computation
1. Extend backward pass to accept vector multiplier
2. Modify gradient accumulation logic for weighted gradients
3. Implement VJP method in Autograd class
4. Add basic unit tests for simple functions

### Phase 2: Batch VJP Support
1. Implement batch vector multiplier handling
2. Add parallel computation for independent vectors
3. Optimize memory usage for batch operations
4. Add batch VJP unit tests

### Phase 3: Optimization Features
1. Implement sparsity exploitation (skip zero entries)
2. Add intermediate gradient caching
3. Implement gradient checkpointing option
4. Add performance profiling and benchmarks

## Testing Requirements

### Correctness Tests
- Test VJP against numerical differentiation for simple functions (linear, quadratic)
- Test VJP for multi-output functions (output dimension > 1)
- Test VJP for functions with complex computation graphs
- Test batch VJP consistency with individual VJP computations

### Edge Cases
- Test with zero vector multiplier
- Test with single-element vector multiplier
- Test with all-zero gradient scenarios
- Test with very large vector multipliers (> 1000 dimensions)

### Performance Tests
- Benchmark VJP vs numerical differentiation for accuracy and speed
- Test memory usage for large computation graphs
- Benchmark batch VJP scaling with number of vectors
- Test sparsity exploitation benefits

## Dependencies
- Extended gradient tape (spec_gradient_tape_extension.md)
- Computation graph infrastructure
- Tensor operations (basic arithmetic, linear algebra)
- Parallel computation framework (for batch operations)

## Success Criteria
- VJP matches numerical differentiation within 1e-6 tolerance for simple functions
- VJP computation time < 10% of numerical differentiation for large functions
- Support vector multipliers up to 10,000 dimensions
- Batch VJP scales linearly with number of vectors
- Memory usage for VJP < 2x forward pass memory

## Notes for Coder
- Reuse existing gradient computation logic as much as possible
- Consider implementing VJP as a special case of backward pass
- Add comprehensive input validation (vector dimensions must match output)
- Document the computation graph traversal algorithm
- Test thread safety for parallel batch VJP computation
- Consider implementing VJP for specific operations (e.g., matmul, conv) with optimized kernels
