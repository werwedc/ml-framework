# Spec: Hessian-Vector Product (HVP) Implementation

## Overview
Implement Hessian-vector product (HVP) computation without materializing the full Hessian matrix. HVP is critical for large-scale optimization and second-order methods where full Hessian computation is infeasible.

## Requirements

### Core Functionality
- Compute HVP: H * v where H is the Hessian and v is a vector
- Efficient computation using nested automatic differentiation
- Support for large-scale models (millions of parameters)
- Enable batch HVP computation for multiple vectors

### API Design
```csharp
// Basic HVP computation
Tensor hvp = Autograd.HessianVectorProduct(loss, parameters, vector);

// Batch HVP for multiple vectors
Tensor[] hvpBatch = Autograd.HessianVectorProduct(loss, parameters, vectorBatch);

// In-place computation with memory efficiency
Autograd.HessianVectorProduct(loss, parameters, vector, outputBuffer);

// HVP with gradient checkpointing (memory efficient for large models)
Tensor hvp = Autograd.HessianVectorProduct(loss, parameters, vector, useCheckpointing: true);
```

### Technical Details

#### Mathematical Foundation
- Given scalar loss function L with parameters p
- Hessian H is n×n matrix: H_ij = ∂²L/∂p_i∂p_j
- HVP computes: H * v where v ∈ R^n
- Can be computed efficiently using nested differentiation:
  1. Compute gradient g = ∇L(p)
  2. Compute directional derivative: H*v = d/dε[g(p + ε*v)] at ε=0

#### Implementation Strategy

**Pearlmutter's Trick (Nested Differentiation)**
1. Compute first gradient: g = ∇L(p)
2. Treat g as a function to be differentiated
3. Compute directional derivative of g in direction v:
   - Set gradient of L as new loss function
   - Set parameters as p, with initial gradients = v
   - Perform backward pass to get HVP

**Algorithm Steps**
```
function HVP(L, p, v):
    # Forward pass
    p.requires_grad = True
    loss = L(p)

    # First backward pass
    grad = gradient(loss, p)  # g = ∇L(p)

    # Set up for second backward pass
    p.grad = v  # Set initial gradients to vector v
    grad.requires_grad = True  # Make gradient differentiable

    # Second backward pass (directional derivative)
    hvp = gradient(sum(grad * v), p)  # H*v

    return hvp
```

#### Memory Optimization Techniques

**Gradient Checkpointing**
- Recompute intermediate activations instead of storing them
- Trade computation for memory
- Critical for large models (e.g., transformers, CNNs)

**Tangent Accumulation**
- Accumulate tangents in place to reduce memory
- Reuse gradient buffers
- Implement efficient tangent propagation

**Chunking**
- Compute HVP in chunks for very large parameter sets
- Process parameter groups sequentially
- Combine results at the end

#### Performance Optimizations
- Reuse forward pass activations if memory allows
- Parallelize independent parameter groups
- Cache intermediate results for repeated HVPs
- Implement fused operations for common patterns

## Implementation Tasks

### Phase 1: Core HVP Computation
1. Implement nested differentiation for HVP
2. Extend gradient tape to support nested backward passes
3. Implement HVP method using Pearlmutter's trick
4. Add basic unit tests for small models

### Phase 2: Memory Optimization
1. Implement gradient checkpointing for HVP
2. Add in-place tangent accumulation
3. Implement parameter chunking for large models
4. Add memory usage tests and benchmarks

### Phase 3: Batch HVP Support
1. Implement batch vector multiplier handling
2. Add parallel HVP computation for independent vectors
3. Optimize memory usage for batch operations
4. Add batch HVP unit tests

### Phase 4: Optimization Features
1. Implement result caching for repeated HVPs
2. Add automatic checkpointing based on memory constraints
3. Implement efficient parallel computation strategies
4. Add performance profiling and benchmarks

## Testing Requirements

### Correctness Tests
- Test HVP against numerical differentiation for simple functions (quadratic)
- Test HVP for complex functions (neural networks)
- Test HVP symmetry: H*v should match v^T*H (same result)
- Test batch HVP consistency with individual HVP computations

### Edge Cases
- Test with zero vector multiplier
- Test with single-element vector multiplier
- Test with all-zero Hessian scenarios (constant loss)
- Test with very large parameter sets (> 1M parameters)

### Performance Tests
- Benchmark HVP vs full Hessian computation for memory usage
- Test HVP computation time for models of various sizes (1K, 10K, 100K, 1M parameters)
- Test memory usage with and without checkpointing
- Benchmark batch HVP scaling with number of vectors

### Large-Scale Tests
- Test HVP on realistic neural network models (MLP, CNN, Transformer)
- Verify memory usage scales linearly, not quadratically, with model size
- Test HVP computation time for 1M parameter model (< 500ms target)
- Test parameter chunking for very large models (> 10M parameters)

## Dependencies
- Extended gradient tape (spec_gradient_tape_extension.md)
- VJP implementation (spec_vjp_implementation.md)
- Memory management system
- Parallel computation framework (for batch operations)

## Success Criteria
- HVP matches numerical differentiation within 1e-6 tolerance for simple functions
- HVP computation time for 1M parameter model < 500ms
- Memory usage for HVP scales linearly with model size (O(n) not O(n²))
- Support models up to 10M parameters with checkpointing
- Batch HVP scales linearly with number of vectors

## Notes for Coder
- Pearlmutter's trick is the standard algorithm - implement this first
- Gradient checkpointing is critical for large models - implement early
- Consider implementing HVP for specific operations (e.g., matmul, conv) with optimized kernels
- Add comprehensive memory profiling and warnings
- Test thread safety for parallel batch HVP computation
- Document the nested differentiation algorithm clearly
- Consider implementing automatic checkpointing threshold based on model size
- Add progress callbacks for long-running HVP computations on large models
