# Spec: Precision Enum and Detection Utilities

## Overview
Define the core precision enumeration and hardware detection utilities that form the foundation for mixed-precision training.

## Dependencies
- None (foundational component)

## Implementation Details

### 1. Precision Enum
Create an enum in `src/MLFramework/Optimizers/MixedPrecision/Precision.cs`:

```csharp
namespace MLFramework.Optimizers.MixedPrecision;

public enum Precision
{
    FP32,  // 32-bit float (default)
    FP16,  // 16-bit float (half precision)
    BF16   // 16-bit bfloat (brain floating point)
}
```

### 2. PrecisionCapability Struct
Create a struct to describe hardware capabilities in `src/MLFramework/Optimizers/MixedPrecision/PrecisionCapability.cs`:

```csharp
namespace MLFramework.Optimizers.MixedPrecision;

public readonly struct PrecisionCapability
{
    public bool SupportsFP16 { get; init; }
    public bool SupportsBF16 { get; init; }
    public bool SupportsFP32 { get; init; }  // Always true

    public bool IsFP16Available => SupportsFP16;
    public bool IsBF16Available => SupportsBF16;
}
```

### 3. HardwareDetector Static Class
Create a static class in `src/MLFramework/Optimizers/MixedPrecision/HardwareDetector.cs`:

```csharp
namespace MLFramework.Optimizers.MixedPrecision;

public static class HardwareDetector
{
    // Cache detection results
    private static PrecisionCapability? _cachedCapability;

    /// <summary>
    /// Detects mixed-precision capabilities of the current hardware
    /// </summary>
    public static PrecisionCapability DetectCapabilities()
    {
        if (_cachedCapability.HasValue)
            return _cachedCapability.Value;

        var capability = new PrecisionCapability
        {
            SupportsFP32 = true,  // Always supported
            SupportsFP16 = CheckFP16Support(),
            SupportsBF16 = CheckBF16Support()
        };

        _cachedCapability = capability;
        return capability;
    }

    /// <summary>
    /// Returns the recommended precision for current hardware
    /// Priority: BF16 > FP16 > FP32
    /// </summary>
    public static Precision GetRecommendedPrecision()
    {
        var capability = DetectCapabilities();
        if (capability.SupportsBF16)
            return Precision.BF16;
        if (capability.SupportsFP16)
            return Precision.FP16;
        return Precision.FP32;
    }

    private static bool CheckFP16Support()
    {
        // TODO: Implement actual hardware detection
        // For now, return false (will be mocked in tests)
        return false;
    }

    private static bool CheckBF16Support()
    {
        // TODO: Implement actual hardware detection
        // For now, return false (will be mocked in tests)
        return false;
    }

    /// <summary>
    /// Clear cached detection (useful for testing)
    /// </summary>
    public static void ResetCache()
    {
        _cachedCapability = null;
    }
}
```

### 4. PrecisionConverter Static Class
Create utility class for precision conversions in `src/MLFramework/Optimizers/MixedPrecision/PrecisionConverter.cs`:

```csharp
namespace MLFramework.Optimizers.MixedPrecision;

public static class PrecisionConverter
{
    /// <summary>
    /// Converts tensor to target precision
    /// </summary>
    public static ITensor Convert(ITensor tensor, Precision targetPrecision)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var currentPrecision = DetectPrecision(tensor);

        if (currentPrecision == targetPrecision)
            return tensor;

        // TODO: Implement actual conversion
        // For now, return tensor as-is (will be implemented in subsequent specs)
        return tensor;
    }

    /// <summary>
    /// Detects the precision of a tensor
    /// </summary>
    public static Precision DetectPrecision(ITensor tensor)
    {
        // TODO: Implement based on tensor dtype
        // For now, return FP32 as default
        return Precision.FP32;
    }

    /// <summary>
    /// Checks if conversion is safe (no significant precision loss)
    /// </summary>
    public static bool IsConversionSafe(Precision from, Precision to)
    {
        // FP32 -> FP16/BF16: May lose precision but generally safe
        // FP16/BF16 -> FP32: Always safe (no loss)
        // FP16 <-> BF16: May have precision differences

        if (from == to)
            return true;

        if (to == Precision.FP32)
            return true;

        // Converting to reduced precision
        return true;  // Conservative: allow all conversions
    }
}
```

## Requirements

### Functional Requirements
1. **Precision Enum**: Must support FP32, FP16, and BF16
2. **Hardware Detection**: Must detect FP16 and BF16 support
3. **Recommendation Logic**: Must recommend best available precision
4. **Caching**: Detection results must be cached
5. **Cache Reset**: Must provide method to clear cache for testing
6. **Thread Safety**: Detection should be thread-safe

### Non-Functional Requirements
1. **Performance**: Detection should be fast (< 1ms)
2. **Memory**: Should have minimal memory footprint
3. **Error Handling**: Handle null tensors gracefully
4. **Extensibility**: Easy to add new precision types

## Deliverables

### Source Files
1. `src/MLFramework/Optimizers/MixedPrecision/Precision.cs`
2. `src/MLFramework/Optimizers/MixedPrecision/PrecisionCapability.cs`
3. `src/MLFramework/Optimizers/MixedPrecision/HardwareDetector.cs`
4. `src/MLFramework/Optimizers/MixedPrecision/PrecisionConverter.cs`

### Unit Tests
- `tests/MLFramework.Tests/Optimizers/MixedPrecision/HardwareDetectorTests.cs`
  - Test detection caching
  - Test recommendation logic
  - Test cache reset
  - Test thread safety

- `tests/MLFramework.Tests/Optimizers/MixedPrecision/PrecisionConverterTests.cs`
  - Test safe conversion checks
  - Test precision detection (mocked)
  - Test null handling

## Notes for Coder
- Hardware detection (CheckFP16Support/CheckBF16Support) should be implemented as stubs for now
- Tensor conversion in PrecisionConverter.Convert should be stubbed for now
- These will be fully implemented in subsequent specs when tensor infrastructure is ready
- Focus on the API design and class structure
- Ensure thread-safety for detection caching (use double-check locking or Lazy<T>)
