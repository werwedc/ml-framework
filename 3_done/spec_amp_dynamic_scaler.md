# Spec: AMP Dynamic Loss Scaler

## Overview
Implement a dynamic loss scaler that automatically adjusts the scaling factor during training based on overflow detection, preventing gradient underflow in FP16 training.

## Class Specification

### 1. DynamicLossScaler Class

**File:** `src/MLFramework/Amp/DynamicLossScaler.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Dynamic loss scaler with automatic adjustment
    /// Prevents gradient underflow in FP16 training by adapting to overflow
    /// </summary>
    public class DynamicLossScaler : ILossScaler
    {
        /// <summary>
        /// Gets the current scaling factor
        /// </summary>
        public float Scale { get; private set; }

        /// <summary>
        /// Gets whether the scaler is enabled
        /// </summary>
        public bool Enabled { get; }

        /// <summary>
        /// Gets the growth factor for increasing scale
        /// </summary>
        public float GrowthFactor { get; }

        /// <summary>
        /// Gets the backoff factor for decreasing scale
        /// </summary>
        public float BackoffFactor { get; }

        /// <summary>
        /// Gets the number of consecutive iterations without overflow before increasing scale
        /// </summary>
        public int GrowthInterval { get; }

        /// <summary>
        /// Gets the minimum allowed scale
        /// </summary>
        public float MinScale { get; }

        /// <summary>
        /// Gets the maximum allowed scale
        /// </summary>
        public float MaxScale { get; }

        /// <summary>
        /// Gets the number of consecutive iterations without overflow
        /// </summary>
        public int GrowthCounter { get; private set; }

        /// <summary>
        /// Gets the total number of overflows encountered
        /// </summary>
        public int TotalOverflows { get; private set; }

        /// <summary>
        /// Creates a new DynamicLossScaler with default parameters
        /// </summary>
        public DynamicLossScaler();

        /// <summary>
        /// Creates a new DynamicLossScaler with custom parameters
        /// </summary>
        /// <param name="initialScale">Initial scaling factor (default: 2^16 = 65536)</param>
        /// <param name="growthFactor">Factor to multiply scale when increasing (default: 2.0)</param>
        /// <param name="backoffFactor">Factor to multiply scale when decreasing (default: 0.5)</param>
        /// <param name="growthInterval">Iterations without overflow before increasing (default: 2000)</param>
        /// <param name="minScale">Minimum allowed scale (default: 1.0)</param>
        /// <param name="maxScale">Maximum allowed scale (default: 2^24 = 16777216)</param>
        /// <param name="enabled">Whether to enable scaling (default: true)</param>
        public DynamicLossScaler(
            float initialScale = 65536.0f,
            float growthFactor = 2.0f,
            float backoffFactor = 0.5f,
            int growthInterval = 2000,
            float minScale = 1.0f,
            float maxScale = 16777216.0f,
            bool enabled = true);

        /// <summary>
        /// Scales the loss tensor before backward pass
        /// </summary>
        /// <param name="loss">The loss tensor to scale</param>
        /// <returns>Scaled loss tensor</returns>
        public Tensor ScaleLoss(Tensor loss);

        /// <summary>
        /// Unscales gradients after backward pass
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>Unscaled gradients</returns>
        public Dictionary<string, Tensor> UnscaleGradients(Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Unscales a single gradient tensor
        /// </summary>
        /// <param name="gradient">The gradient tensor to unscale</param>
        /// <returns>Unscaled gradient</returns>
        public Tensor UnscaleGradient(Tensor gradient);

        /// <summary>
        /// Checks for overflow in gradients
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>True if overflow detected, false otherwise</returns>
        public bool CheckOverflow(Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Checks for overflow in a single gradient tensor
        /// </summary>
        /// <param name="gradient">The gradient tensor to check</param>
        /// <returns>True if overflow detected, false otherwise</returns>
        public bool CheckOverflow(Tensor gradient);

        /// <summary>
        /// Updates the scale factor based on overflow detection
        /// </summary>
        /// <param name="overflow">Whether overflow was detected</param>
        public void UpdateScale(bool overflow);

        /// <summary>
        /// Gets the scale as a tensor for loss multiplication
        /// </summary>
        /// <returns>Scale value as a scalar tensor</returns>
        public Tensor GetScaleTensor();

        /// <summary>
        /// Gets the inverse scale for gradient unscaling
        /// </summary>
        /// <returns>Inverse scale as a scalar tensor</returns>
        public Tensor GetInverseScaleTensor();

        /// <summary>
        /// Resets the scaler to initial state
        /// </summary>
        public void Reset();

        /// <summary>
        /// Gets statistics about the scaler's performance
        /// </summary>
        public DynamicScalerStats GetStats();
    }
}
```

### 2. DynamicScalerStats Class

**File:** `src/MLFramework/Amp/DynamicScalerStats.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Statistics for the dynamic loss scaler
    /// </summary>
    public class DynamicScalerStats
    {
        /// <summary>
        /// Gets the current scale factor
        /// </summary>
        public float CurrentScale { get; }

        /// <summary>
        /// Gets the total number of overflows
        /// </summary>
        public int TotalOverflows { get; }

        /// <summary>
        /// Gets the total number of iterations without overflow
        /// </summary>
        public int TotalSuccessfulIterations { get; }

        /// <summary>
        /// Gets the number of times the scale was increased
        /// </summary>
        public int ScaleIncreaseCount { get; }

        /// <summary>
        /// Gets the number of times the scale was decreased
        /// </summary>
        public int ScaleDecreaseCount { get; }

        /// <summary>
        /// Gets the minimum scale reached
        /// </summary>
        public float MinScaleReached { get; }

        /// <summary>
        /// Gets the maximum scale reached
        /// </summary>
        public float MaxScaleReached { get; }

        /// <summary>
        /// Gets the success rate (iterations without overflow / total iterations)
        /// </summary>
        public float SuccessRate { get; }

        /// <summary>
        /// Creates a new DynamicScalerStats
        /// </summary>
        public DynamicScalerStats(
            float currentScale,
            int totalOverflows,
            int totalSuccessfulIterations,
            int scaleIncreaseCount,
            int scaleDecreaseCount,
            float minScaleReached,
            float maxScaleReached);

        /// <summary>
        /// Returns a string representation of the statistics
        /// </summary>
        public override string ToString();
    }
}
```

### 3. DynamicScalerConfig Class

**File:** `src/MLFramework/Amp/DynamicScalerConfig.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Configuration for dynamic loss scaler
    /// </summary>
    public class DynamicScalerConfig
    {
        /// <summary>
        /// Gets or sets the initial scale factor
        /// </summary>
        public float InitialScale { get; set; }

        /// <summary>
        /// Gets or sets the growth factor for increasing scale
        /// </summary>
        public float GrowthFactor { get; set; }

        /// <summary>
        /// Gets or sets the backoff factor for decreasing scale
        /// </summary>
        public float BackoffFactor { get; set; }

        /// <summary>
        /// Gets or sets the growth interval
        /// </summary>
        public int GrowthInterval { get; set; }

        /// <summary>
        /// Gets or sets the minimum scale
        /// </summary>
        public float MinScale { get; set; }

        /// <summary>
        /// Gets or sets the maximum scale
        /// </summary>
        public float MaxScale { get; set; }

        /// <summary>
        /// Creates a default configuration
        /// </summary>
        public static DynamicScalerConfig CreateDefault();

        /// <summary>
        /// Creates a conservative configuration (slower scale increase)
        /// </summary>
        public static DynamicScalerConfig CreateConservative();

        /// <summary>
        /// Creates an aggressive configuration (faster scale increase)
        /// </summary>
        public static DynamicScalerConfig CreateAggressive();
    }
}
```

## Implementation Details

### UpdateScale Algorithm

```csharp
public void UpdateScale(bool overflow)
{
    if (overflow)
    {
        // Decrease scale by backoff factor
        Scale = Math.Max(Scale * BackoffFactor, MinScale);
        GrowthCounter = 0;
        TotalOverflows++;
    }
    else
    {
        // Increment growth counter
        GrowthCounter++;

        // Increase scale after growthInterval consecutive successes
        if (GrowthCounter >= GrowthInterval)
        {
            Scale = Math.Min(Scale * GrowthFactor, MaxScale);
            GrowthCounter = 0;
        }
    }
}
```

### Overflow Detection
- Same as StaticLossScaler (check for Inf/NaN)
- Early exit on first detected overflow

### Scale Factor Constraints
- `MinScale` prevents underflow (default: 1.0)
- `MaxScale` prevents explosion (default: 2^24)
- Scale is clamped after each update

### Statistics Tracking
- Track min/max scale reached during training
- Count scale increases/decreases
- Calculate success rate

## Usage Example

```csharp
// Create dynamic scaler with custom parameters
var scaler = new DynamicLossScaler(
    initialScale: 65536.0f,
    growthFactor: 2.0f,
    backoffFactor: 0.5f,
    growthInterval: 2000
);

// Training loop
for (int epoch = 0; epoch < epochs; epoch++)
{
    var loss = model.Forward(inputs);
    loss = scaler.ScaleLoss(loss);
    loss.Backward();

    var grads = model.GetGradients();
    bool hasOverflow = scaler.CheckOverflow(grads);

    // Update scale before optimizer step
    scaler.UpdateScale(hasOverflow);

    if (hasOverflow)
    {
        Console.WriteLine($"Overflow detected, skipping step (scale: {scaler.Scale})");
        continue;
    }

    var unscaledGrads = scaler.UnscaleGradients(grads);
    optimizer.Step(unscaledGrads);

    // Print stats every 100 iterations
    if (iteration % 100 == 0)
    {
        var stats = scaler.GetStats();
        Console.WriteLine($"Scale: {stats.CurrentScale}, Overflows: {stats.TotalOverflows}");
    }
}
```

## Dependencies
- MLFramework.Core (Tensor, DataType)
- MLFramework.Amp (ILossScaler interface)
- System (Math, DateTime for stats)

## Testing Requirements
- Test scale increase after growthInterval
- Test scale decrease on overflow
- Test scale clamping (min/max bounds)
- Test statistics tracking accuracy
- Test reset functionality
- Test with various growth factors and backoff factors
- Test with different growth intervals
- Test edge cases (overflow on first iteration, continuous overflow, etc.)

## Success Criteria
- [ ] Scale increases correctly after growthInterval
- [ ] Scale decreases correctly on overflow
- [ ] Scale stays within min/max bounds
- [ ] Statistics are tracked accurately
- [ ] Reset functionality works correctly
- [ ] Performance overhead < 10% of total training time
- [ ] All unit tests pass
- [ ] Documentation is complete with usage examples
