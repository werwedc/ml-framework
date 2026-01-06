# Spec: Autograd Tests Suite

## Overview
Implement comprehensive test suite for the autograd system, including gradient checking utilities, regression tests, and performance benchmarks.

## Files to Create
- `tests/MLFramework.Tests/Autograd/GradientChecker.cs`
- `tests/MLFramework.Tests/Autograd/AutogradRegressionTests.cs`
- `tests/MLFramework.Tests/Autograd/AutogradPerformanceTests.cs`
- `tests/MLFramework.Tests/Autograd/AutogradIntegrationTests.cs`

## API Design

### Class: GradientChecker
```csharp
public static class GradientChecker
{
    public static bool CheckGradient(Func<Tensor, Tensor> f, Tensor x, double tolerance = 1e-6);
    public static bool CheckGradients(Func<Tensor[], Tensor> f, Tensor[] inputs, double tolerance = 1e-6);
    public static double ComputeRelativeError(Tensor analytical, Tensor numerical);
    public static Tensor NumericalGradient(Func<Tensor, Tensor> f, Tensor x, double epsilon = 1e-6);

    // Get detailed gradient error information
    public static GradientCheckResult CheckGradientDetailed(Func<Tensor, Tensor> f, Tensor x, double tolerance = 1e-6);
}

public class GradientCheckResult
{
    public bool Passed { get; }
    public double MaxError { get; }
    public double MeanError { get; }
    public Tensor ErrorMap { get; }
    public string Message { get; }
}
```

### Test Categories

### 1. Unit Tests (`AutogradUnitTests.cs`)
```csharp
[TestFixture]
public class AutogradUnitTests
{
    // Tensor Gradient Tracking
    [Test] public void TestTensorRequiresGrad()
    [Test] public void TestTensorGradStorage()
    [Test] public void TestTensorZeroGrad()
    [Test] public void TestTensorAccumulateGrad()
    [Test] public void TestTensorDetachGrad()

    // Computational Graph
    [Test] public void TestGraphConstruction()
    [Test] public void TestGraphConnections()
    [Test] public void TestGraphClearing()
    [Test] public void TestScopeManagement()

    // Backward Pass
    [Test] public void TestBackwardPassSimple()
    [Test] public void TestBackwardPassChain()
    [Test] public void TestBackwardPassBranching()
    [Test] public void TestGraphRetention()

    // Operation Gradients
    [Test] public void TestAddGradient()
    [Test] public void TestMulGradient()
    [Test] public void TestDivGradient()
    [Test] public void TestPowGradient()
    [Test] public void TestReluGradient()
    [Test] public void TestSigmoidGradient()
    [Test] public void TestTanhGradient()
    [Test] public void TestSumGradient()
    [Test] public void TestMeanGradient()

    // Custom Functions
    [Test] public void TestCustomAutogradFunction()
    [Test] public void TestFunctionRegistry()
}
```

### 2. Integration Tests (`AutogradIntegrationTests.cs`)
```csharp
[TestFixture]
public class AutogradIntegrationTests
{
    // Neural Network Components
    [Test] public void TestLinearLayerGradients()
    [Test] public void TestConv2DLayerGradients()
    [Test] public void TestMaxPoolGradients()
    [Test] public void TestBatchNormGradients()

    // Training Scenarios
    [Test] public void TestSimpleRegressionTraining()
    [Test] public void TestBinaryClassificationTraining()
    [Test] public void TestMultiClassClassificationTraining()
    [Test] public void TestResidualNetworkGradients()
    [Test] public void TestLSTMGradients()

    // Gradient Accumulation
    [Test] public void TestGradientAccumulation4Steps()
    [Test] public void TestGradientAccumulationScaling()

    // Gradient Checkpointing
    [Test] public void TestManualCheckpointing()
    [Test] public void TestAutoCheckpointing()
    [Test] public void TestCheckpointMemorySavings()

    // Higher-Order Derivatives
    [Test] public void TestJacobianComputation()
    [Test] public void TestHessianComputation()
    [Test] public void TestGradientOfGradient()

    // Edge Cases
    [Test] public void TestZeroGradient()
    [Test] public void TestNaNPropagation()
    [Test] public void TestInfPropagation()
    [Test] public void TestNumericalStability()
}
```

### 3. Regression Tests (`AutogradRegressionTests.cs`)
```csharp
[TestFixture]
public class AutogradRegressionTests
{
    // Known issues / fixed bugs
    [Test] public void Regression_BroadcastGradient()
    [Test] public void Regression_MultipleGradientPaths()
    [Test] public void Regression_InPlaceOperationGradients()
    [Test] public void Regression_ViewsAndSlices()
    [Test] public void Regression_DetachedTensorGradients()

    // Compatibility tests (compare with PyTorch/TensorFlow)
    [Test] public void CompareWithPyTorch_SimpleNetwork()
    [Test] public void CompareWithPyTorch_ComplexOperation()
}
```

### 4. Performance Tests (`AutogradPerformanceTests.cs`)
```csharp
[TestFixture]
public class AutogradPerformanceTests
{
    [Test] public void BenchmarkForwardPass_Overhead()
    [Test] public void BenchmarkBackwardPass_Speed()
    [Test] public void BenchmarkGradientMemory_Usage()
    [Test] public void BenchmarkCheckpointing_Overhead()
    [Test] public void BenchmarkJacobianComputation_Time()

    // Large scale tests
    [Test] public void TestLargeModelGradients_1MParameters()
    [Test] public void TestDeepNetwork_100Layers()
}
```

## Test Utilities

### Gradient Checking Helper
```csharp
public static class AutogradTestHelpers
{
    public static Tensor CreateRandomTensor(int[] shape, bool requiresGrad = true, double mean = 0.0, double std = 1.0);
    public static void AssertTensorEqual(Tensor a, Tensor b, double tolerance = 1e-6);
    public static void AssertGradientsClose(Func<Tensor, Tensor> f, Tensor x, double tolerance = 1e-6);
    public static double ComputeMemoryUsage(Action action);

    // PyTorch comparison (if reference implementation available)
    public static void CompareWithPyTorch(string pytorchCode, Tensor expected);
}
```

## Testing Requirements

### Coverage Goals
- **Line Coverage**: > 90%
- **Branch Coverage**: > 85%
- **Function Coverage**: 100%

### Test Categories
1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test components working together
3. **Regression Tests**: Ensure fixed bugs don't reappear
4. **Performance Tests**: Benchmark key operations

### Gradient Accuracy
- Compare analytical gradients with numerical gradients
- Use finite difference method for reference
- Tolerance: 1e-6 for simple operations, 1e-4 for complex operations
- Test with various tensor shapes and data distributions

### Edge Cases
- Zero gradients
- NaN/Inf values
- Broadcasting edge cases
- Empty tensors
- Very large tensors
- Very small tensors
- Non-differentiable operations

### Memory Testing
- Verify memory cleanup after backward pass
- Test memory leaks with multiple iterations
- Verify gradient checkpointing memory savings
- Test with GPU memory (if available)

## Implementation Notes

### Test Organization
- Follow NUnit/xUnit conventions
- Use setup/teardown for common test infrastructure
- Parameterized tests for multiple shapes/types
- Test data fixtures for reproducibility

### Continuous Integration
- All tests must pass before merging
- Performance tests should have acceptable thresholds
- Memory tests should have clear limits
- Regression tests should never fail

### Test Data Management
- Use seeded random generators for reproducibility
- Store test data separately from implementation
- Version test data with code

## Dependencies
- NUnit or xUnit framework
- All autograd components
- Core tensor operations
- BenchmarkDotNet for performance tests

## Success Criteria
- All unit tests pass
- All integration tests pass
- Performance within acceptable ranges
- > 90% code coverage
- No memory leaks detected
- Gradient accuracy validated against numerical gradients
