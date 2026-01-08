# Spec: Higher-Order Derivatives Unit Tests

## Overview
Implement comprehensive unit tests for the higher-order derivatives implementation. Tests should cover correctness, edge cases, performance, and numerical stability across all derivative types.

## Requirements

### Testing Scope
- Correctness tests for all derivative computations
- Edge case testing (zero gradients, special structures)
- Numerical stability testing
- Performance benchmarking
- Integration tests with realistic models
- Memory usage testing

### Test Structure

#### Test Organization
```
tests/
├── HigherOrderDerivatives/
│   ├── GradientTapeExtensionTests.cs
│   ├── VJPTests.cs
│   ├── JVPTests.cs
│   ├── JacobianTests.cs
│   ├── HVPTests.cs
│   ├── HessianTests.cs
│   ├── APITests.cs
│   ├── PerformanceTests.cs
│   └── IntegrationTests.cs
```

#### Common Test Utilities
```csharp
// Numerical differentiation utilities
public static class NumericalDifferentiation
{
    public static Tensor Gradient(Func<Tensor, Tensor> f, Tensor x, double epsilon = 1e-6);
    public static Tensor Jacobian(Func<Tensor, Tensor> f, Tensor x);
    public static Tensor Hessian(Func<Tensor, Tensor> f, Tensor x);
    public static bool IsEqual(Tensor a, Tensor b, double tolerance = 1e-6);
}

// Test data generators
public static class TestDataGenerator
{
    public static (Func<Tensor, Tensor>, Tensor) GenerateQuadraticFunction();
    public static (Func<Tensor, Tensor>, Tensor) GenerateSinusoidalFunction();
    public static (Func<Tensor, Tensor>, Tensor) GenerateComplexNeuralNetwork();
}

// Performance measurement utilities
public static class PerformanceProfiler
{
    public static ProfilerResult ProfileComputation(Action computation);
    public static MemorySnapshot CaptureMemoryUsage();
}
```

## Test Cases

### GradientTapeExtensionTests

```csharp
[TestClass]
public class GradientTapeExtensionTests
{
    [TestMethod]
    public void EnableHigherOrderTracking_CreatesTapeWithHigherOrderSupport()
    {
        // Test that higher-order tracking is enabled
    }

    [TestMethod]
    public void NestedTapes_ComputeThirdOrderDerivativeCorrectly()
    {
        // Test gradient of gradient of gradient
    }

    [TestMethod]
    public void GradientRetention_KeepsGradientsWhenRequired()
    {
        // Test gradient retention policy
    }

    [TestMethod]
    public void MemoryUsage_NoLeaksWithNestedTapes()
    {
        // Verify no memory leaks
    }

    [TestMethod]
    public void GraphMerging_CombinesNestedTapesCorrectly()
    {
        // Test graph merging logic
    }
}
```

### VJPTests

```csharp
[TestClass]
public class VJPTests
{
    [TestMethod]
    public void VJP_MatchesNumericalDifferentiation_ForQuadraticFunction()
    {
        // Compare VJP with numerical gradient
    }

    [TestMethod]
    public void VJP_ComputesCorrectGradient_ForMultiOutputFunction()
    {
        // Test with output dimension > 1
    }

    [TestMethod]
    public void BatchVVP_MatchesIndividualVJP_Computations()
    {
        // Test batch vs individual computations
    }

    [TestMethod]
    public void VVP_WithZeroVector_ReturnsZeroGradient()
    {
        // Test edge case
    }

    [TestMethod]
    public void VVP_HandlesSparseVectors_Efficiently()
    {
        // Test sparsity exploitation
    }
}
```

### JVPTests

```csharp
[TestClass]
public class JVPTests
{
    [TestMethod]
    public void JVP_MatchesNumericalDifferentiation_ForQuadraticFunction()
    {
        // Compare JVP with numerical gradient
    }

    [TestMethod]
    public void JVP_ComputesCorrectDerivative_ForMultiInputFunction()
    {
        // Test with input dimension > 1
    }

    [TestMethod]
    public void BatchJVP_MatchesIndividualJVP_Computations()
    {
        // Test batch vs individual computations
    }

    [TestMethod]
    public void JVP_WithZeroVector_ReturnsZeroTangent()
    {
        // Test edge case
    }

    [TestMethod]
    public void JVP_IsFasterThanVVP_WhenInputDimSmallerThanOutputDim()
    {
        // Test mode efficiency
    }
}
```

### JacobianTests

```csharp
[TestClass]
public class JacobianTests
{
    [TestMethod]
    public void Jacobian_MatchesNumericalDifferentiation_ForSimpleFunctions()
    {
        // Test Jacobian correctness
    }

    [TestMethod]
    public void Jacobian_IsSymmetric_WhenAppropriate()
    {
        // Test Jacobian properties
    }

    [TestMethod]
    public void AutoMode_SelectsOptimalStrategy_BasedOnDimensions()
    {
        // Test automatic mode selection
    }

    [TestMethod]
    public void SparseJacobian_MatchesDenseJacobian_ForSameFunction()
    {
        // Test sparse vs dense consistency
    }

    [TestMethod]
    public void Jacobian_DetectsDiagonalStructure_Correctly()
    {
        // Test structure detection
    }

    [TestMethod]
    public void Jacobian_ComputesPartialJacobian_Correctly()
    {
        // Test partial Jacobian computation
    }
}
```

### HVPTests

```csharp
[TestClass]
public class HVPTests
{
    [TestMethod]
    public void HVP_MatchesNumericalDifferentiation_ForQuadraticFunction()
    {
        // Test HVP correctness
    }

    [TestMethod]
    public void HVP_IsSymmetric_HVP_v_Equals_vT_H()
    {
        // Test Hessian symmetry property
    }

    [TestMethod]
    public void HVP_WithCheckpointing_SavesMemory_LargeModel()
    {
        // Test memory efficiency
    }

    [TestMethod]
    public void HVP_ScalesLinearly_WithModelSize()
    {
        // Test scaling properties
    }

    [TestMethod]
    public void HVP_ComputesCorrectly_ForLargeModel_1MParameters()
    {
        // Test large-scale computation
    }

    [TestMethod]
    public void BatchHVP_MatchesIndividualHVP_Computations()
    {
        // Test batch vs individual computations
    }
}
```

### HessianTests

```csharp
[TestClass]
public class HessianTests
{
    [TestMethod]
    public void Hessian_MatchesNumericalDifferentiation_ForQuadraticFunction()
    {
        // Test Hessian correctness
    }

    [TestMethod]
    public void Hessian_IsSymmetric()
    {
        // Test Hessian symmetry
    }

    [TestMethod]
    public void Hessian_ComputesEigenvalues_Correctly()
    {
        // Test eigenvalue computation
    }

    [TestMethod]
    public void SparseHessian_MatchesDenseHessian_ForSameFunction()
    {
        // Test sparse vs dense consistency
    }

    [TestMethod]
    public void Hessian_DetectsDiagonalStructure_Correctly()
    {
        // Test structure detection
    }

    [TestMethod]
    public void Hessian_ComputesConditionNumber_Correctly()
    {
        // Test Hessian analysis utilities
    }
}
```

### APITests

```csharp
[TestClass]
public class APITests
{
    [TestMethod]
    public void JacobianAPI_WorksCorrectly_ForSimpleFunctions()
    {
        // Test API usability
    }

    [TestMethod]
    public void HessianAPI_WorksCorrectly_ForSimpleFunctions()
    {
        // Test API usability
    }

    [TestMethod]
    public void ArbitraryOrderDerivative_ComputesCorrectly()
    {
        // Test nth-order derivative API
    }

    [TestMethod]
    public void ContextBasedAPI_WorksCorrectly_ForNestedDifferentiation()
    {
        // Test tape-based API
    }

    [TestMethod]
    public void API_ThrowsHelpfulException_ForInvalidInput()
    {
        // Test error handling
    }

    [TestMethod]
    public void Validation_CorrectlyIdentifiesNonDifferentiableOps()
    {
        // Test validation utilities
    }
}
```

### PerformanceTests

```csharp
[TestClass]
public class PerformanceTests
{
    [TestMethod]
    public void Jacobian100Dim_ComputesInUnder100ms()
    {
        // Performance benchmark
    }

    [TestMethod]
    public void HVP_1MParameters_ComputesInUnder500ms()
    {
        // Performance benchmark
    }

    [TestMethod]
    public void FourthOrderDerivative_ComputesFor10KParams()
    {
        // Higher-order performance
    }

    [TestMethod]
    public void SparseJacobian_IsFasterThanDense_ForSparseProblem()
    {
        // Sparsity performance
    }

    [TestMethod]
    public void MemoryUsage_HVP_ScalesLinearly()
    {
        // Memory scaling test
    }
}
```

### IntegrationTests

```csharp
[TestClass]
public class IntegrationTests
{
    [TestMethod]
    public void MAML_ComputesGradientsOfGradients_Correctly()
    {
        // Test meta-learning use case
    }

    [TestMethod]
    public void NewtonMethod_UsesHessian_Correctly()
    {
        // Test optimization use case
    }

    [TestMethod]
    public void NeuralODE_ComputesHigherOrderDerivatives_Correctly()
    {
        // Test differential equation use case
    }

    [TestMethod]
    public void API_IntegratesWithExistingOptimizers()
    {
        // Test optimizer integration
    }

    [TestMethod]
    public void Differentiation_WorksWithAllModelTypes()
    {
        // Test with MLP, CNN, RNN, Transformer
    }
}
```

## Implementation Tasks

### Phase 1: Test Infrastructure
1. Implement test utilities (NumericalDifferentiation, TestDataGenerator, PerformanceProfiler)
2. Set up test project structure
3. Implement base test class with common setup/teardown

### Phase 2: Core Correctness Tests
1. Implement GradientTapeExtensionTests
2. Implement VJPTests
3. Implement JVPTests
4. Implement JacobianTests
5. Implement HVPTests
6. Implement HessianTests

### Phase 3: Performance and Integration Tests
1. Implement PerformanceTests with benchmarks
2. Implement IntegrationTests with real use cases
3. Implement APITests for usability and validation

### Phase 4: Test Automation
1. Set up continuous integration testing
2. Add performance regression detection
3. Add code coverage reporting

## Testing Requirements

### Coverage Targets
- > 90% code coverage for all derivative computations
- > 95% coverage for API surface
- All edge cases tested
- All error paths tested

### Performance Baselines
- Jacobian (100 dim): < 100ms
- HVP (1M params): < 500ms
- 4th-order derivative (10K params): < 5s
- Memory usage scales linearly for HVP

### Stability Requirements
- All tests pass consistently (no flaky tests)
- Numerical precision within specified tolerances
- No memory leaks in stress tests

## Dependencies
- All implementation specs (must be implemented first)
- Test framework (NUnit, xUnit, or MSTest)
- Benchmarking framework (BenchmarkDotNet)
- Memory profiling tools

## Success Criteria
- All tests pass consistently
- Code coverage > 90%
- Performance benchmarks meet targets
- Integration tests cover real use cases
- Test suite runs in reasonable time (< 10 minutes)

## Notes for Coder
- Use descriptive test names that explain what is being tested
- Implement test data generators for reproducible tests
- Add assertions with meaningful error messages
- Test both happy paths and error cases
- Use parameterized tests for testing multiple scenarios
- Add comments explaining complex test setups
- Keep tests independent and fast where possible
- Use setup/teardown for common test infrastructure
- Consider property-based testing for numerical correctness
- Add performance regression detection in CI
