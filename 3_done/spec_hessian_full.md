# Spec: Full Hessian Computation

## Overview
Implement full Hessian matrix computation for scalar functions. While infeasible for very large models, full Hessians are valuable for small parameter sets, debugging, and analysis.

## Requirements

### Core Functionality
- Compute full Hessian matrix H for scalar loss functions
- Support both dense and sparse Hessian representations
- Efficient computation for small parameter sets (< 10K parameters)
- Automatic structure detection (diagonal, block-diagonal, etc.)

### API Design
```csharp
// Basic Hessian computation (dense)
Tensor hessian = Autograd.Hessian(loss, parameters);

// Sparse Hessian computation
SparseTensor sparseHessian = Autograd.Hessian(loss, parameters, sparse: true);

// Hessian with structure detection
Tensor hessian = Autograd.Hessian(loss, parameters, detectStructure: true);

// Hessian with eigenvalue computation
(Tensor hessian, Tensor eigenvalues) = Autograd.HessianWithEigenvalues(loss, parameters);

// Partial Hessian (subset of parameters)
Tensor partialHessian = Autograd.Hessian(loss, parameters, parameterIndices: new[] {0, 2, 5, 10});
```

### Technical Details

#### Hessian Computation Strategies

**Strategy 1: Column-by-Column using HVP**
- For each parameter dimension i, compute HVP with standard basis vector e_i
- Assemble HVP results as columns of Hessian
- Complexity: O(n) HVP computations for n parameters
- Uses efficient HVP implementation from spec_hvp_implementation.md

**Strategy 2: Direct Double Differentiation**
- Compute gradient g = ∇L(p)
- Differentiate each component of g with respect to all parameters
- More straightforward but potentially less efficient
- Better for very small problems

**Strategy 3: Analytical Computation for Specific Operations**
- Derive Hessian formulas for common operations
- Implement specialized kernels for each operation
- Most efficient when applicable
- Requires extensive operation-specific implementations

#### Symmetry Exploitation
- Hessian is symmetric: H_ij = H_ji
- Only compute upper or lower triangular portion
- Mirror to complete full Hessian
- Reduces computation by ~50%

#### Sparse Hessian Representation
- Identify zero entries in Hessian efficiently
- Use compressed sparse row (CSR) format
- Exploit sparsity in neural network Hessians (often highly sparse)
- Support sparse linear algebra operations

#### Structure Detection
- Detect diagonal Hessians (element-wise loss functions)
- Detect block-diagonal structures (layer-wise independence)
- Detect banded structures (in recurrent networks)
- Use specialized storage and computation for detected structures

#### Eigenvalue Computation
- Compute eigenvalues for Hessian analysis
- Use iterative methods (Lanczos, power iteration) for large Hessians
- Compute full eigendecomposition for small Hessians
- Support eigenvalue-based analysis (condition number, definiteness)

## Implementation Tasks

### Phase 1: Core Hessian Computation
1. Implement Hessian method using column-by-column HVP strategy
2. Add symmetry exploitation (compute only triangular portion)
3. Implement full Hessian assembly
4. Add basic unit tests for small Hessians

### Phase 2: Sparse Hessian Support
1. Implement zero entry detection algorithm
2. Add sparse Hessian representation (CSR format)
3. Implement sparse Hessian computation
4. Add sparse Hessian unit tests

### Phase 3: Structure Detection
1. Implement diagonal Hessian detection
2. Implement block-diagonal detection
3. Implement specialized storage and computation for structures
4. Add performance benchmarks for structured problems

### Phase 4: Eigenvalue Computation
1. Implement full eigendecomposition for small Hessians
2. Implement iterative methods (Lanczos) for large Hessians
3. Add HessianWithEigenvalues method
4. Add eigenvalue-based analysis utilities

## Testing Requirements

### Correctness Tests
- Test Hessian against numerical differentiation for simple functions (quadratic)
- Test Hessian symmetry: H[i,j] should equal H[j,i]
- Test sparse Hessian correctness against dense Hessian
- Test structure detection accuracy

### Edge Cases
- Test with single-parameter functions (1x1 Hessian)
- Test with zero Hessian (constant loss)
- Test with identity Hessian (quadratic loss)
- Test with very small Hessians (2x2, 3x3)

### Performance Tests
- Benchmark HVP-based vs direct double differentiation
- Test memory usage for dense vs sparse Hessians
- Benchmark structure detection and specialized computation
- Test scaling with parameter count (100, 1000, 10000)

### Numerical Stability Tests
- Test Hessian for ill-conditioned problems
- Test eigenvalue computation accuracy
- Test numerical precision with different floating-point types (float32, float64)
- Test Hessian for functions with non-differentiable regions

## Dependencies
- HVP implementation (spec_hvp_implementation.md)
- Extended gradient tape (spec_gradient_tape_extension.md)
- Sparse tensor infrastructure (if not already exists)
- Linear algebra utilities (for eigendecomposition)

## Success Criteria
- Full Hessian matches numerical differentiation within 1e-6 tolerance
- Support Hessians up to 1000x1000 dimensions in < 1 second
- Symmetry exploitation reduces computation time by ~40-50%
- Sparse Hessian computation > 5x faster than dense for 80%+ sparse
- Eigenvalue computation accurate to 1e-6 for small Hessians

## Notes for Coder
- Reuse HVP implementation - don't reimplement nested differentiation
- Symmetry exploitation is easy and provides significant speedup - implement early
- Consider implementing progressive computation with early termination for analysis
- Add warnings for potentially expensive computations (e.g., 10K x 10K Hessian)
- Document the HVP-based algorithm clearly
- Test numerical stability extensively - Hessians can amplify numerical errors
- Consider implementing Hessian-vector products for linear algebra operations (solving linear systems)
- Add Hessian analysis utilities: condition number, definiteness checking, spectral analysis
