# Spec: AMP Static Loss Scaler

## Overview
Implement a static loss scaler that maintains a constant scaling factor to prevent gradient underflow in FP16 training.

## Class Specification

### 1. StaticLossScaler Class

**File:** `src/MLFramework/Amp/StaticLossScaler.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Static loss scaler with a constant scaling factor
    /// Prevents gradient underflow in FP16 training
    /// </summary>
    public class StaticLossScaler : ILossScaler
    {
        /// <summary>
        /// Gets the current scaling factor
        /// </summary>
        public float Scale { get; }

        /// <summary>
        /// Gets whether the scaler is enabled
        /// </summary>
        public bool Enabled { get; }

        /// <summary>
        /// Creates a new StaticLossScaler
        /// </summary>
        /// <param name="scale">Constant scaling factor (default: 2^16 = 65536)</param>
        /// <param name="enabled">Whether to enable scaling (default: true)</param>
        public StaticLossScaler(float scale = 65536.0f, bool enabled = true);

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
        /// Updates the scale (no-op for static scaler)
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
    }
}
```

### 2. ILossScaler Interface

**File:** `src/MLFramework/Amp/ILossScaler.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Interface for loss scaling implementations
    /// </summary>
    public interface ILossScaler
    {
        /// <summary>
        /// Gets the current scaling factor
        /// </summary>
        float Scale { get; }

        /// <summary>
        /// Gets whether the scaler is enabled
        /// </summary>
        bool Enabled { get; }

        /// <summary>
        /// Scales the loss tensor before backward pass
        /// </summary>
        /// <param name="loss">The loss tensor to scale</param>
        /// <returns>Scaled loss tensor</returns>
        Tensor ScaleLoss(Tensor loss);

        /// <summary>
        /// Unscales gradients after backward pass
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>Unscaled gradients</returns>
        Dictionary<string, Tensor> UnscaleGradients(Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Checks for overflow in gradients
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>True if overflow detected, false otherwise</returns>
        bool CheckOverflow(Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Updates the scale based on overflow detection
        /// </summary>
        /// <param name="overflow">Whether overflow was detected</param>
        void UpdateScale(bool overflow);
    }
}
```

### 3. ScaleFactor Helper Class

**File:** `src/MLFramework/Amp/ScaleFactor.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Utility class for common scale factor values
    /// </summary>
    public static class ScaleFactor
    {
        /// <summary>
        /// No scaling (scale = 1.0)
        /// </summary>
        public const float None = 1.0f;

        /// <summary>
        /// Conservative scale (2^8 = 256)
        /// </summary>
        public const float Conservative = 256.0f;

        /// <summary>
        /// Moderate scale (2^16 = 65536)
        /// </summary>
        public const float Moderate = 65536.0f;

        /// <summary>
        /// Aggressive scale (2^20 = 1048576)
        /// </summary>
        public const float Aggressive = 1048576.0f;

        /// <summary>
        /// Creates a power-of-two scale factor
        /// </summary>
        /// <param name="exponent">The exponent (scale = 2^exponent)</param>
        /// <returns>The scale factor</returns>
        public static float PowerOfTwo(int exponent);

        /// <summary>
        /// Gets the recommended scale for a given precision
        /// </summary>
        /// <param name="precision">The target precision</param>
        /// <returns>Recommended scale factor</returns>
        public static float GetRecommendedScale(DataType precision);
    }
}
```

## Implementation Details

### Loss Scaling Logic
1. **Scale Loss**: Multiply loss by `scale` before backward pass
2. **Backward Pass**: Compute gradients with scaled loss
3. **Unscale Gradients**: Divide gradients by `scale` after backward pass
4. **Overflow Check**: Check for Inf/NaN in unscaled gradients

### Overflow Detection
- Check each gradient tensor for Inf or NaN values
- Early exit on first detected overflow
- Use efficient tensor operations (any() for boolean check)

### Tensor Operations
```csharp
// Scale loss
scaled_loss = loss * scale_tensor

// Unscale gradients
unscaled_grad = grad / scale_tensor

// Check overflow
has_overflow = tensor.IsInf() || tensor.IsNaN()
```

### Performance Considerations
- Cache scale and inverse scale as scalar tensors
- Use in-place operations where possible
- Avoid redundant overflow checks

## Usage Example

```csharp
// Create static scaler
var scaler = new StaticLossScaler(scale: 65536.0f);

// Training loop
for (int epoch = 0; epoch < epochs; epoch++)
{
    var loss = model.Forward(inputs);
    loss = scaler.ScaleLoss(loss);
    loss.Backward();

    var grads = model.GetGradients();
    if (scaler.CheckOverflow(grads))
    {
        Console.WriteLine("Overflow detected, skipping update");
        continue;
    }

    var unscaledGrads = scaler.UnscaleGradients(grads);
    optimizer.Step(unscaledGrads);
}
```

## Dependencies
- MLFramework.Core (Tensor, DataType)
- MLFramework.Amp (ILossScaler interface)
- System.Collections.Generic (Dictionary)

## Testing Requirements
- Test loss scaling with various scale factors
- Test gradient unscaling accuracy
- Test overflow detection with Inf/NaN gradients
- Test edge cases (scale = 0, scale = Inf, etc.)
- Test disabled scaler (no scaling applied)
- Test with different dtypes (FP16, BF16, FP32)

## Success Criteria
- [ ] Loss is scaled correctly by the scale factor
- [ ] Gradients are unscaled accurately
- [ ] Overflow detection works for Inf and NaN
- [ ] Disabled scaler passes through unchanged values
- [ ] Performance overhead < 5% of total training time
- [ ] All unit tests pass
- [ ] Documentation is complete with usage examples
