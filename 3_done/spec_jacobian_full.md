# Spec: Full Jacobian Computation

## Overview
Implement full Jacobian matrix computation by combining VJP and JVP operations. The implementation will automatically select the most efficient strategy based on input/output dimensions.

## Requirements

### Core Functionality
- Compute full Jacobian matrix J for vector-valued functions
- Automatic mode selection (forward vs reverse) based on dimensions
- Support for both dense and sparse Jacobian representations
- Efficient computation for special cases (diagonal, triangular, etc.)

### API Design
```csharp
// Basic Jacobian computation (dense)
Tensor jacobian = Autograd.Jacobian(function, parameters);

// Sparse Jacobian computation
SparseTensor sparseJacobian = Autograd.Jacobian(function, parameters, sparse: true);

// Specified mode (force forward or reverse mode)
Tensor jacobian = Autograd.Jacobian(function, parameters, mode: JacobianMode.Auto);
Tensor jacobian = Autograd.Jacobian(function, parameters, mode: JacobianMode.Forward);
Tensor jacobian = Autograd.Jacobian(function, parameters, mode: JacobianMode.Reverse);

// Jacobian for specific output indices (partial Jacobian)
Tensor partialJacobian = Autograd.Jacobian(function, parameters, outputIndices: new[] {0, 2, 5});
```

### Technical Details

#### Jacobian Computation Strategies

**Strategy 1: Column-by-Column using VJP**
- For each output dimension i, compute VJP with standard basis vector e_i
- Assemble VJP results as columns of Jacobian
- Complexity: O(m) backward passes for m output dimensions
- Best when m << n (output dim < input dim)

**Strategy 2: Row-by-Row using JVP**
- For each input dimension j, compute JVP with standard basis vector e_j
- Assemble JVP results as rows of Jacobian
- Complexity: O(n) forward passes for n input dimensions
- Best when n << m (input dim < output dim)

**Strategy 3: Hybrid Approach**
- Use mixed strategy for large problems
- Compute blocks of Jacobian using optimal method for each block
- Balance between forward and reverse passes

#### Automatic Mode Selection
- Estimate computational cost for each strategy
- Select strategy with lower estimated cost
- Consider memory constraints and available parallelism
- Provide mode override for expert users

#### Sparse Jacobian Representation
- Identify zero entries in Jacobian efficiently
- Use compressed sparse row (CSR) or column (CSC) format
- Exploit structure in computation graph (e.g., diagonal operations)
- Support sparse arithmetic operations

#### Special Structure Detection
- Detect diagonal Jacobians (element-wise operations)
- Detect block-diagonal structures
- Detect triangular structures (e.g., in recurrent networks)
- Use specialized algorithms for detected structures

## Implementation Tasks

### Phase 1: Core Jacobian Computation
1. Implement Jacobian method with automatic mode selection
2. Implement column-by-column VJP strategy
3. Implement row-by-row JVP strategy
4. Add basic unit tests for small Jacobians

### Phase 2: Automatic Mode Selection
1. Implement cost estimation for forward and reverse modes
2. Add automatic mode selection logic
3. Implement mode override functionality
4. Add tests for mode selection correctness

### Phase 3: Sparse Jacobian Support
1. Implement zero entry detection algorithm
2. Add sparse tensor representation (CSR/CSC)
3. Implement sparse Jacobian computation
4. Add sparse Jacobian unit tests

### Phase 4: Special Structure Detection
1. Implement diagonal Jacobian detection
2. Implement block-diagonal detection
3. Implement specialized algorithms for detected structures
4. Add performance benchmarks for structured problems

## Testing Requirements

### Correctness Tests
- Test Jacobian against numerical differentiation for simple functions
- Test Jacobian for multi-dimensional functions (both n, m > 1)
- Test automatic mode selection picks optimal strategy
- Test sparse Jacobian correctness against dense Jacobian
- Test special structure detection and handling

### Edge Cases
- Test with single-input single-output functions (1x1 Jacobian)
- Test with zero Jacobian (constant function)
- Test with identity Jacobian (linear function)
- Test with very high-dimensional functions (> 10K dimensions)

### Performance Tests
- Benchmark automatic mode selection vs fixed modes
- Test memory usage for large dense Jacobians
- Benchmark sparse Jacobian vs dense Jacobian for sparse problems
- Test scaling with problem size for different strategies

## Dependencies
- VJP implementation (spec_vjp_implementation.md)
- JVP implementation (spec_jvp_implementation.md)
- Extended gradient tape (spec_gradient_tape_extension.md)
- Sparse tensor infrastructure (if not already exists)

## Success Criteria
- Full Jacobian matches numerical differentiation within 1e-6 tolerance
- Automatic mode selection achieves near-optimal performance
- Support Jacobians up to 100x100 dimensions in < 1 second
- Sparse Jacobian computation > 10x faster than dense for 90%+ sparse
- Memory usage scales appropriately with Jacobian sparsity

## Notes for Coder
- Reuse VJP and JVP implementations - don't reimplement
- Consider implementing parallel computation for independent columns/rows
- Add progress callbacks for long-running Jacobian computations
- Document the mode selection criteria for users
- Test numerical stability for ill-conditioned Jacobians
- Consider implementing Jacobian caching for repeated computations
- Add warnings for potentially expensive computations (e.g., 10K x 10K Jacobian)
