# Spec: AMP Unit Tests

## Overview
Comprehensive unit tests for the Automatic Mixed Precision (AMP) system, covering all components from datatypes to integrations.

## Test File Organization

### Test Structure
```
tests/
├── MLFramework.Amp.Tests/
│   ├── Datatypes/
│   │   ├── HalfTests.cs
│   │   ├── BFloat16Tests.cs
│   │   └── AmpCastTests.cs
│   ├── Core/
│   │   ├── DataTypeExtensionsTests.cs
│   │   ├── DataTypeInfoTests.cs
│   │   └── AmpConfigTests.cs
│   ├── Registry/
│   │   ├── OpPrecisionRuleTests.cs
│   │   ├── AmpRegistryTests.cs
│   │   └── DefaultAmpRulesTests.cs
│   ├── Scalers/
│   │   ├── StaticLossScalerTests.cs
│   │   ├── DynamicLossScalerTests.cs
│   │   ├── ScaleFactorTests.cs
│   │   └── DynamicScalerStatsTests.cs
│   ├── Api/
│   │   ├── GradScalerTests.cs
│   │   ├── GradScalerFactoryTests.cs
│   │   └── GradScalerContextTests.cs
│   ├── AutoCast/
│   │   ├── AutoCastTests.cs
│   │   ├── AutoCastContextTests.cs
│   │   └── AmpEnabledAttributeTests.cs
│   ├── Utils/
│   │   ├── GradientUtilsTests.cs
│   │   ├── OverflowStatsTests.cs
│   │   └── GradientClipperTests.cs
│   ├── Kernel/
│   │   ├── KernelSelectorTests.cs
│   │   ├── KernelCapabilityTests.cs
│   │   ├── KernelPerformanceStatsTests.cs
│   │   └── KernelRegistryTests.cs
│   ├── Integrations/
│   │   ├── AmpAutogradContextTests.cs
│   │   ├── AmpAutogradFunctionTests.cs
│   │   ├── AmpTensorExtensionsTests.cs
│   │   ├── AmpAutogradHelperTests.cs
│   │   ├── AmpCustomFunctionTests.cs
│   │   ├── AmpOptimizerWrapperTests.cs
│   │   ├── AmpOptimizerHelperTests.cs
│   │   └── AmpOptimizerExtensionsTests.cs
│   └── Integration/
│       ├── EndToEndTrainingTests.cs
│       ├── PerformanceBenchmarks.cs
│       └── AccuracyTests.cs
```

## Detailed Test Specifications

### 1. Half Tests

**File:** `tests/MLFramework.Amp.Tests/Datatypes/HalfTests.cs`

```csharp
public class HalfTests
{
    [Fact]
    public void Constructor_FromFloat_PreservesValue()
    {
        // Test that float to Half conversion preserves value
    }

    [Fact]
    public void Arithmetic_Addition_WorksCorrectly()
    {
        // Test Half addition
    }

    [Fact]
    public void Arithmetic_Multiplication_WorksCorrectly()
    {
        // Test Half multiplication
    }

    [Fact]
    public void Comparison_Equality_WorksCorrectly()
    {
        // Test Half equality
    }

    [Fact]
    public void SpecialValues_NaN_HandledCorrectly()
    {
        // Test NaN handling
    }

    [Fact]
    public void SpecialValues_Inf_HandledCorrectly()
    {
        // Test Infinity handling
    }

    [Theory]
    [InlineData(0.0f)]
    [InlineData(1.0f)]
    [InlineData(100.0f)]
    [InlineData(-100.0f)]
    public void Conversion_FloatToHalf_WithinTolerance(float value)
    {
        // Test float to Half conversion accuracy
    }

    [Fact]
    public void Constants_MaxValue_IsCorrect()
    {
        // Test Half.MaxValue = 65504
    }
}
```

### 2. BFloat16 Tests

**File:** `tests/MLFramework.Amp.Tests/Datatypes/BFloat16Tests.cs`

```csharp
public class BFloat16Tests
{
    [Fact]
    public void Constructor_FromFloat_PreservesValue()
    {
        // Test that float to BFloat16 conversion preserves value
    }

    [Fact]
    public void Arithmetic_Addition_WorksCorrectly()
    {
        // Test BFloat16 addition
    }

    [Fact]
    public void Comparison_LessThan_WorksCorrectly()
    {
        // Test BFloat16 comparison
    }

    [Fact]
    public void SpecialValues_NaN_HandledCorrectly()
    {
        // Test NaN handling
    }

    [Theory]
    [InlineData(1.0f)]
    [InlineData(100.0f)]
    [InlineData(1e-5f)]
    [InlineData(1e5f)]
    public void Conversion_FloatToBFloat16_WithinTolerance(float value)
    {
        // Test float to BFloat16 conversion accuracy
    }
}
```

### 3. DataType Extensions Tests

**File:** `tests/MLFramework.Amp.Tests/Core/DataTypeExtensionsTests.cs`

```csharp
public class DataTypeExtensionsTests
{
    [Fact]
    public void GetSize_Float16_ReturnsTwo()
    {
        Assert.Equal(2, DataType.Float16.GetSize());
    }

    [Fact]
    public void IsFloatType_BFloat16_ReturnsTrue()
    {
        Assert.True(DataType.BFloat16.IsFloatType());
    }

    [Fact]
    public void IsLowPrecision_Float16_ReturnsTrue()
    {
        Assert.True(DataType.Float16.IsLowPrecision());
    }

    [Fact]
    public void GetHigherPrecision_Float16_ReturnsFloat32()
    {
        Assert.Equal(DataType.Float32, DataType.Float16.GetHigherPrecision());
    }

    [Fact]
    public void GetLowerPrecision_Float32_ReturnsBFloat16()
    {
        Assert.Equal(DataType.BFloat16, DataType.Float32.GetLowerPrecision());
    }
}
```

### 4. Amp Registry Tests

**File:** `tests/MLFramework.Amp.Tests/Registry/AmpRegistryTests.cs`

```csharp
public class AmpRegistryTests
{
    private readonly AmpConfig _config = AmpConfig.CreateBf16();

    [Fact]
    public void RegisterWhitelist_AddsOperation()
    {
        // Test whitelist registration
    }

    [Fact]
    public void RegisterBlacklist_AddsOperation()
    {
        // Test blacklist registration
    }

    [Fact]
    public void GetRule_ReturnsCorrectRule()
    {
        // Test rule retrieval
    }

    [Fact]
    public void GetForwardDtype_Whitelisted_ReturnsBFloat16()
    {
        // Test dtype selection for whitelisted operation
    }

    [Fact]
    public void GetForwardDtype_Blacklisted_ReturnsFloat32()
    {
        // Test dtype selection for blacklisted operation
    }

    [Fact]
    public void IsWhitelisted_ReturnsCorrectValue()
    {
        // Test whitelist checking
    }

    [Fact]
    public void Unregister_RemovesRule()
    {
        // Test rule removal
    }

    [Fact]
    public void Clear_RemovesAllRules()
    {
        // Test clearing all rules
    }
}
```

### 5. Static Loss Scaler Tests

**File:** `tests/MLFramework.Amp.Tests/Scalers/StaticLossScalerTests.cs`

```csharp
public class StaticLossScalerTests
{
    [Fact]
    public void ScaleLoss_MultipliesByScale()
    {
        // Test loss scaling
    }

    [Fact]
    public void UnscaleGradients_DividesByScale()
    {
        // Test gradient unscaling
    }

    [Fact]
    public void CheckOverflow_WithInf_ReturnsTrue()
    {
        // Test overflow detection with Inf
    }

    [Fact]
    public void CheckOverflow_WithNaN_ReturnsTrue()
    {
        // Test overflow detection with NaN
    }

    [Fact]
    public void UpdateScale_StaticScaler_NoChange()
    {
        // Test that static scaler doesn't change scale
    }

    [Fact]
    public void Disabled_Scaler_PassesThrough()
    {
        // Test disabled scaler passes values unchanged
    }

    [Theory]
    [InlineData(1.0f)]
    [InlineData(65536.0f)]
    [InlineData(1048576.0f)]
    public void ScaleLoss_WithVariousScales_WorksCorrectly(float scale)
    {
        // Test loss scaling with various scales
    }
}
```

### 6. Dynamic Loss Scaler Tests

**File:** `tests/MLFramework.Amp.Tests/Scalers/DynamicLossScalerTests.cs`

```csharp
public class DynamicLossScalerTests
{
    [Fact]
    public void UpdateScale_Overflow_DecreasesScale()
    {
        // Test scale decrease on overflow
    }

    [Fact]
    public void UpdateScale_NoOverflow_IncreasesAfterInterval()
    {
        // Test scale increase after growth interval
    }

    [Fact]
    public void UpdateScale_RespectsMinScale()
    {
        // Test min scale constraint
    }

    [Fact]
    public void UpdateScale_RespectsMaxScale()
    {
        // Test max scale constraint
    }

    [Fact]
    public void GrowthCounter_IncreasesOnNoOverflow()
    {
        // Test growth counter increment
    }

    [Fact]
    public void GrowthCounter_ResetsOnOverflow()
    {
        // Test growth counter reset
    }

    [Fact]
    public void GetStats_ReturnsCorrectStatistics()
    {
        // Test statistics tracking
    }

    [Fact]
    public void Reset_RestoresInitialState()
    {
        // Test reset functionality
    }
}
```

### 7. GradScaler API Tests

**File:** `tests/MLFramework.Amp.Tests/Api/GradScalerTests.cs`

```csharp
public class GradScalerTests
{
    [Fact]
    public void Scale_ScalesLossCorrectly()
    {
        // Test loss scaling
    }

    [Fact]
    public void Step_WithOverflow_ReturnsFalse()
    {
        // Test step with overflow detection
    }

    [Fact]
    public void Step_WithoutOverflow_PerformsUpdate()
    {
        // Test step without overflow
    }

    [Fact]
    public void Unscale_ReturnsCorrectGradients()
    {
        // Test gradient unscaling
    }

    [Fact]
    public void Update_CallsUnderlyingScaler()
    {
        // Test update method
    }

    [Fact]
    public void Enable_DisablesWhenCalled()
    {
        // Test enable/disable
    }

    [Fact]
    public void GetStats_ReturnsCorrectStats()
    {
        // Test stats retrieval
    }
}
```

### 8. AutoCast Tests

**File:** `tests/MLFramework.Amp.Tests/AutoCast/AutoCastTests.cs`

```csharp
public class AutoCastTests
{
    [Fact]
    public void Cast_WhitelistedOperation_CastsToBFloat16()
    {
        // Test casting for whitelisted operation
    }

    [Fact]
    public void Cast_BlacklistedOperation_CastsToFloat32()
    {
        // Test casting for blacklisted operation
    }

    [Fact]
    public void GetForwardDtype_ReturnsCorrectDtype()
    {
        // Test forward dtype selection
    }

    [Fact]
    public void GetBackwardDtype_ReturnsCorrectDtype()
    {
        // Test backward dtype selection
    }

    [Fact]
    public void Enter_SetsCurrentContext()
    {
        // Test context enter
    }

    [Fact]
    public void Exit_RestoresPreviousContext()
    {
        // Test context exit
    }

    [Fact]
    public void NestedContexts_WorkCorrectly()
    {
        // Test nested context handling
    }

    [Fact]
    public void Disabled_Cast_ReturnsOriginal()
    {
        // Test disabled AutoCast
    }
}
```

### 9. Gradient Utils Tests

**File:** `tests/MLFramework.Amp.Tests/Utils/GradientUtilsTests.cs`

```csharp
public class GradientUtilsTests
{
    [Fact]
    public void Unscale_DividesByScale()
    {
        // Test gradient unscaling
    }

    [Fact]
    public void CheckOverflow_WithInf_ReturnsTrue()
    {
        // Test Inf detection
    }

    [Fact]
    public void CheckOverflow_WithNaN_ReturnsTrue()
    {
        // Test NaN detection
    }

    [Fact]
    public void CheckOverflow_MultipleGradients_EarlyExit()
    {
        // Test early exit on overflow
    }

    [Fact]
    public void FindOverflowGradients_ReturnsCorrectList()
    {
        // Test finding overflow gradients
    }

    [Fact]
    public void GetOverflowStats_ReturnsCorrectStats()
    {
        // Test overflow statistics
    }
}
```

### 10. Kernel Selector Tests

**File:** `tests/MLFramework.Amp.Tests/Kernel/KernelSelectorTests.cs`

```csharp
public class KernelSelectorTests
{
    [Fact]
    public void GetKernelDtype_SingleTensor_ReturnsCorrectDtype()
    {
        // Test dtype selection for single tensor
    }

    [Fact]
    public void GetKernelDtype_MultipleTensors_SameDtype_ReturnsThatDtype()
    {
        // Test dtype selection for multiple tensors with same dtype
    }

    [Fact]
    public void GetKernelDtype_MultipleTensors_MixedDtype_ReturnsMixed()
    {
        // Test dtype selection for mixed tensors
    }

    [Fact]
    public void IsKernelAvailable_ReturnsCorrectValue()
    {
        // Test kernel availability checking
    }

    [Fact]
    public void SelectBestKernel_PrefersPreferredDtype()
    {
        // Test preferred dtype selection
    }

    [Fact]
    public void SelectBestKernel_FallsBackToFloat32()
    {
        // test fallback to Float32
    }

    [Fact]
    public void UpdatePerformanceStats_TracksCorrectly()
    {
        // Test performance tracking
    }
}
```

### 11. Integration Tests

**File:** `tests/MLFramework.Amp.Tests/Integration/EndToEndTrainingTests.cs`

```csharp
public class EndToEndTrainingTests
{
    [Fact]
    public void TrainingLoop_BasicAMP_WorksCorrectly()
    {
        // Test complete training loop with AMP
    }

    [Fact]
    public void TrainingLoop_DynamicScaler_AdjustsScale()
    {
        // Test dynamic scaler in training loop
    }

    [Fact]
    public void TrainingLoop_OverflowHandling_SkipsCorrectly()
    {
        // Test overflow handling in training
    }

    [Fact]
    public void AutoCast_WithModel_ProducesCorrectOutputs()
    {
        // Test AutoCast with model
    }

    [Fact]
    public void AmpOptimizer_Integration_WorksCorrectly()
    {
        // Test AMP optimizer integration
    }
}
```

## Test Coverage Requirements

### Coverage Targets
- **Datatypes**: 100% coverage (critical for correctness)
- **Core Extensions**: 95% coverage
- **Registry**: 95% coverage
- **Scalers**: 95% coverage
- **API**: 95% coverage
- **AutoCast**: 90% coverage
- **Utils**: 90% coverage
- **Kernel Selector**: 85% coverage
- **Integrations**: 85% coverage
- **Integration Tests**: Full workflow coverage

### Edge Cases to Test
1. Zero values
2. Very large values
3. Very small values
4. NaN values
5. Inf values
6. Scale = 0
7. Scale = Inf
8. Empty gradients
9. Single element tensors
10. Very large tensors
11. Thread safety
12. Multiple overflow scenarios
13. Boundary conditions

## Performance Benchmarks

**File:** `tests/MLFramework.Amp.Tests/Integration/PerformanceBenchmarks.cs`

```csharp
public class PerformanceBenchmarks
{
    [Benchmark]
    public void Half_Arithmetic_Operations()
    {
        // Benchmark Half arithmetic
    }

    [Benchmark]
    public void BFloat16_Arithmetic_Operations()
    {
        // Benchmark BFloat16 arithmetic
    }

    [Benchmark]
    public void StaticLossScaler_ScaleLoss()
    {
        // Benchmark static loss scaling
    }

    [Benchmark]
    public void DynamicLossScaler_UpdateScale()
    {
        // Benchmark dynamic scale update
    }

    [Benchmark]
    public void AutoCast_CastTensor()
    {
        // Benchmark AutoCast
    }

    [Benchmark]
    public void KernelSelector_SelectKernel()
    {
        // Benchmark kernel selection
    }

    [Benchmark]
    public void FullTrainingLoop_WithAMP()
    {
        // Benchmark complete AMP training
    }
}
```

## Test Dependencies

### Required Test Frameworks
- **xUnit**: Unit testing framework
- **xUnit.asserts**: Assertion library
- **Moq**: Mocking framework (for integrations)
- **FluentAssertions**: Fluent assertion library
- **BenchmarkDotNet**: Performance benchmarking

### Test Data Fixtures

**File:** `tests/MLFramework.Amp.Tests/Fixtures/AmpTestFixture.cs`

```csharp
public class AmpTestFixture : IDisposable
{
    public Device Device { get; }
    public AmpConfig Config { get; }
    public AmpRegistry Registry { get; }
    public GradScaler Scaler { get; }
    public Random Random { get; }

    public AmpTestFixture()
    {
        // Setup test fixtures
    }

    public Tensor CreateRandomTensor(int[] shape, DataType dtype)
    {
        // Helper to create test tensors
    }

    public void Dispose()
    {
        // Cleanup
    }
}
```

## Success Criteria

### Test Quality
- [ ] All unit tests pass
- [ ] Test coverage > 90% for all modules
- [ ] All edge cases covered
- [ ] Thread safety verified
- [ ] Performance benchmarks run successfully

### Test Execution
- [ ] Tests execute in < 5 minutes
- [ ] No flaky tests
- [ ] Clear test failure messages
- [ ] Proper test organization

### Integration Tests
- [ ] End-to-end training passes
- [ ] Accuracy matches FP32 within tolerance
- [ ] Performance improvements verified
- [ ] Overflow handling works correctly

### Documentation
- [ ] Test cases are self-documenting
- [ ] Comments explain complex scenarios
- [ ] Test fixtures are well-documented
- [ ] Benchmark results are documented
