# Spec: MixedPrecisionOptions Configuration Class

## Overview
Implement the configuration class that stores all settings for mixed-precision training behavior.

## Dependencies
- Spec 001: Precision enum

## Implementation Details

### MixedPrecisionOptions Class
Create the configuration class in `src/MLFramework/Optimizers/MixedPrecision/MixedPrecisionOptions.cs`:

```csharp
using System;

namespace MLFramework.Optimizers.MixedPrecision;

/// <summary>
/// Configuration options for mixed-precision training optimizer
/// </summary>
public class MixedPrecisionOptions
{
    #region Precision Settings

    /// <summary>
    /// Target precision for training (default: FP16)
    /// </summary>
    public Precision Precision { get; set; } = Precision.FP16;

    /// <summary>
    /// Automatically detect and use best available precision (default: true)
    /// </summary>
    public bool AutoDetectPrecision { get; set; } = true;

    #endregion

    #region Loss Scaling Settings

    /// <summary>
    /// Initial loss scale factor (default: 2^16 = 65536)
    /// </summary>
    public float InitialLossScale { get; set; } = 65536.0f;

    /// <summary>
    /// Factor to multiply loss scale when no overflow occurs (default: 2.0)
    /// </summary>
    public float GrowthFactor { get; set; } = 2.0f;

    /// <summary>
    /// Factor to divide loss scale when overflow occurs (default: 0.5)
    /// </summary>
    public float BackoffFactor { get; set; } = 0.5f;

    /// <summary>
    /// Maximum allowed loss scale to prevent overflow (default: 1e9)
    /// </summary>
    public float MaxLossScale { get; set; } = 1e9f;

    /// <summary>
    /// Minimum allowed loss scale (default: 1.0)
    /// </summary>
    public float MinLossScale { get; set; } = 1.0f;

    /// <summary>
    /// Number of consecutive steps without overflow before growing scale (default: 2000)
    /// </summary>
    public int GrowthInterval { get; set; } = 2000;

    /// <summary>
    /// Whether to enable dynamic loss scaling (default: true)
    /// </summary>
    public bool EnableDynamicLossScaling { get; set; } = true;

    #endregion

    #region Layer Exclusion Settings

    /// <summary>
    /// Automatically keep sensitive layers in FP32 (default: true)
    /// </summary>
    public bool AutoExcludeSensitiveLayers { get; set; } = true;

    /// <summary>
    /// Custom layer types to always keep in FP32
    /// </summary>
    public List<string> Fp32LayerPatterns { get; set; } = new List<string>
    {
        "BatchNorm", "LayerNorm", "InstanceNorm", "GroupNorm"
    };

    #endregion

    #region Gradient Settings

    /// <summary>
    /// Maximum gradient norm for clipping (0 = no clipping, default: 1.0)
    /// </summary>
    public float MaxGradNorm { get; set; } = 1.0f;

    /// <summary>
    /// Whether to clip gradients before optimizer update (default: true)
    /// </summary>
    public bool EnableGradientClipping { get; set; } = true;

    #endregion

    #region Fallback Settings

    /// <summary>
    /// Number of consecutive overflows before disabling mixed precision (default: 10)
    /// </summary>
    public int MaxConsecutiveOverflows { get; set; } = 10;

    /// <summary>
    /// Whether to automatically fall back to FP32 if instability detected (default: true)
    /// </summary>
    public bool EnableAutoFallback { get; set; } = true;

    /// <summary>
    /// Whether to log fallback events (default: true)
    /// </summary>
    public bool LogFallbackEvents { get; set; } = true;

    #endregion

    #region Performance Settings

    /// <summary>
    /// Whether to enable performance monitoring (default: false)
    /// </summary>
    public bool EnablePerformanceMonitoring { get; set; } = false;

    /// <summary>
    /// How often to log performance stats (in steps, default: 100)
    /// </summary>
    public int PerformanceLogInterval { get; set; } = 100;

    #endregion

    #region Validation

    /// <summary>
    /// Validates all options and throws ArgumentException if invalid
    /// </summary>
    public void Validate()
    {
        // Validate loss scaling settings
        if (InitialLossScale <= 0)
            throw new ArgumentException("InitialLossScale must be positive", nameof(InitialLossScale));

        if (GrowthFactor <= 1.0f)
            throw new ArgumentException("GrowthFactor must be > 1.0", nameof(GrowthFactor));

        if (BackoffFactor <= 0 || BackoffFactor >= 1.0f)
            throw new ArgumentException("BackoffFactor must be in (0, 1)", nameof(BackoffFactor));

        if (MaxLossScale <= MinLossScale)
            throw new ArgumentException("MaxLossScale must be > MinLossScale", nameof(MaxLossScale));

        if (GrowthInterval <= 0)
            throw new ArgumentException("GrowthInterval must be positive", nameof(GrowthInterval));

        // Validate gradient settings
        if (MaxGradNorm < 0)
            throw new ArgumentException("MaxGradNorm must be non-negative", nameof(MaxGradNorm));

        // Validate fallback settings
        if (MaxConsecutiveOverflows <= 0)
            throw new ArgumentException("MaxConsecutiveOverflows must be positive", nameof(MaxConsecutiveOverflows));

        // Validate performance settings
        if (PerformanceLogInterval <= 0)
            throw new ArgumentException("PerformanceLogInterval must be positive", nameof(PerformanceLogInterval));
    }

    /// <summary>
    /// Creates a deep copy of these options
    /// </summary>
    public MixedPrecisionOptions Clone()
    {
        var clone = new MixedPrecisionOptions
        {
            Precision = Precision,
            AutoDetectPrecision = AutoDetectPrecision,
            InitialLossScale = InitialLossScale,
            GrowthFactor = GrowthFactor,
            BackoffFactor = BackoffFactor,
            MaxLossScale = MaxLossScale,
            MinLossScale = MinLossScale,
            GrowthInterval = GrowthInterval,
            EnableDynamicLossScaling = EnableDynamicLossScaling,
            AutoExcludeSensitiveLayers = AutoExcludeSensitiveLayers,
            MaxGradNorm = MaxGradNorm,
            EnableGradientClipping = EnableGradientClipping,
            MaxConsecutiveOverflows = MaxConsecutiveOverflows,
            EnableAutoFallback = EnableAutoFallback,
            LogFallbackEvents = LogFallbackEvents,
            EnablePerformanceMonitoring = EnablePerformanceMonitoring,
            PerformanceLogInterval = PerformanceLogInterval
        };

        // Deep copy the layer patterns list
        clone.Fp32LayerPatterns = new List<string>(Fp32LayerPatterns);
        return clone;
    }

    #endregion

    #region Factory Methods

    /// <summary>
    /// Creates options optimized for FP16 training
    /// </summary>
    public static MixedPrecisionOptions ForFP16()
    {
        return new MixedPrecisionOptions
        {
            Precision = Precision.FP16,
            AutoDetectPrecision = false,
            InitialLossScale = 65536.0f,
            GrowthFactor = 2.0f,
            BackoffFactor = 0.5f
        };
    }

    /// <summary>
    /// Creates options optimized for BF16 training
    /// BF16 typically requires less aggressive loss scaling
    /// </summary>
    public static MixedPrecisionOptions ForBF16()
    {
        return new MixedPrecisionOptions
        {
            Precision = Precision.BF16,
            AutoDetectPrecision = false,
            InitialLossScale = 1.0f,  // BF16 has wider dynamic range
            GrowthFactor = 2.0f,
            BackoffFactor = 0.5f,
            GrowthInterval = 1000
        };
    }

    /// <summary>
    /// Creates conservative options for sensitive models
    /// </summary>
    public static MixedPrecisionOptions Conservative()
    {
        return new MixedPrecisionOptions
        {
            Precision = Precision.FP16,
            AutoDetectPrecision = false,
            InitialLossScale = 8192.0f,  // Lower initial scale
            GrowthFactor = 1.5f,  // Slower growth
            BackoffFactor = 0.75f,  // Slower backoff
            GrowthInterval = 4000,  // Longer interval
            AutoExcludeSensitiveLayers = true,
            MaxGradNorm = 0.5f,  // Stricter clipping
            MaxConsecutiveOverflows = 5  // Faster fallback
        };
    }

    #endregion
}
```

## Requirements

### Functional Requirements
1. **Configuration Storage**: Must store all mixed-precision training settings
2. **Validation**: Must validate all settings on demand
3. **Cloning**: Must provide deep copy functionality
4. **Factory Methods**: Must provide common presets (FP16, BF16, Conservative)
5. **Layer Patterns**: Must support custom FP32 layer exclusion patterns
6. **Loss Scaling**: Must support dynamic loss scaling configuration

### Non-Functional Requirements
1. **Immutability**: Factory methods should return new instances
2. **Type Safety**: Use strong types throughout
3. **Documentation**: All public properties must have XML documentation
4. **Validation Coverage**: All settings must be validated

## Deliverables

### Source Files
1. `src/MLFramework/Optimizers/MixedPrecision/MixedPrecisionOptions.cs`

### Unit Tests
- `tests/MLFramework.Tests/Optimizers/MixedPrecision/MixedPrecisionOptionsTests.cs`
  - Test validation with valid options
  - Test validation with invalid options (throw exceptions)
  - Test clone creates independent copy
  - Test factory methods return valid options
  - Test layer patterns list independence

## Notes for Coder
- Ensure all validation checks throw ArgumentException with descriptive messages
- Factory methods should call Validate() before returning (or validate in tests)
- Make sure Clone() creates a truly independent copy (deep copy lists)
- Keep default values aligned with common practice from PyTorch AMP
- XML documentation should describe the impact of each setting
