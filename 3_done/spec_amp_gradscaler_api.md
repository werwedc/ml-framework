# Spec: AMP GradScaler API

## Overview
Create a user-friendly GradScaler API that wraps the loss scaler implementations and provides a high-level interface for training with mixed precision.

## Class Specification

### 1. GradScaler Class

**File:** `src/MLFramework/Amp/GradScaler.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// High-level GradScaler API for training with mixed precision
    /// Wraps ILossScaler with convenience methods
    /// </summary>
    public class GradScaler
    {
        private readonly ILossScaler _scaler;

        /// <summary>
        /// Gets the underlying loss scaler
        /// </summary>
        public ILossScaler Scaler => _scaler;

        /// <summary>
        /// Gets the current scaling factor
        /// </summary>
        public float Scale => _scaler.Scale;

        /// <summary>
        /// Gets whether the scaler is enabled
        /// </summary>
        public bool Enabled => _scaler.Enabled;

        /// <summary>
        /// Creates a GradScaler with a static loss scaler
        /// </summary>
        /// <param name="scale">Constant scaling factor (default: 2^16 = 65536)</param>
        /// <param name="enabled">Whether to enable scaling (default: true)</param>
        public GradScaler(float scale = 65536.0f, bool enabled = true);

        /// <summary>
        /// Creates a GradScaler with a dynamic loss scaler
        /// </summary>
        /// <param name="initialScale">Initial scaling factor</param>
        /// <param name="growthFactor">Factor to multiply scale when increasing</param>
        /// <param name="backoffFactor">Factor to multiply scale when decreasing</param>
        /// <param name="growthInterval">Iterations without overflow before increasing</param>
        /// <param name="minScale">Minimum allowed scale</param>
        /// <param name="maxScale">Maximum allowed scale</param>
        /// <param name="enabled">Whether to enable scaling</param>
        public GradScaler(
            float initialScale = 65536.0f,
            float growthFactor = 2.0f,
            float backoffFactor = 0.5f,
            int growthInterval = 2000,
            float minScale = 1.0f,
            float maxScale = 16777216.0f,
            bool enabled = true);

        /// <summary>
        /// Creates a GradScaler with a custom ILossScaler implementation
        /// </summary>
        /// <param name="scaler">The loss scaler to wrap</param>
        public GradScaler(ILossScaler scaler);

        /// <summary>
        /// Scales the loss tensor before backward pass
        /// </summary>
        /// <param name="loss">The loss tensor to scale</param>
        /// <returns>Scaled loss tensor</returns>
        public Tensor Scale(Tensor loss);

        /// <summary>
        /// Unscales gradients and prepares them for optimizer step
        /// </summary>
        /// <param name="optimizer">The optimizer to get gradients from</param>
        /// <param name="optimizerStep">Whether to call optimizer.Step() after unscaling</param>
        /// <returns>True if optimizer step was performed, false if skipped</returns>
        public bool Step(IOptimizer optimizer, bool optimizerStep = true);

        /// <summary>
        /// Unscales gradients manually (for custom optimizer logic)
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>Unscaled gradients</returns>
        public Dictionary<string, Tensor> Unscale(Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Updates the scale factor (for dynamic scalers)
        /// </summary>
        public void Update();

        /// <summary>
        /// Checks for overflow in gradients
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>True if overflow detected, false otherwise</returns>
        public bool CheckOverflow(Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Enables the scaler
        /// </summary>
        public void Enable();

        /// <summary>
        /// Disables the scaler
        /// </summary>
        public void Disable();

        /// <summary>
        /// Resets the scaler (for dynamic scalers)
        /// </summary>
        public void Reset();

        /// <summary>
        /// Gets the scale as a tensor for loss multiplication
        /// </summary>
        /// <returns>Scale value as a scalar tensor</returns>
        public Tensor GetScaleTensor();

        /// <summary>
        /// Gets statistics (for dynamic scalers)
        /// </summary>
        /// <returns>Scaler statistics if available, null otherwise</returns>
        public DynamicScalerStats? GetStats();
    }
}
```

### 2. GradScalerFactory Class

**File:** `src/MLFramework/Amp/GradScalerFactory.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Factory for creating GradScaler instances with common configurations
    /// </summary>
    public static class GradScalerFactory
    {
        /// <summary>
        /// Creates a default dynamic loss scaler (recommended for most use cases)
        /// </summary>
        public static GradScaler CreateDefault();

        /// <summary>
        /// Creates a static loss scaler with moderate scale
        /// </summary>
        public static GradScaler CreateStatic();

        /// <summary>
        /// Creates a static loss scaler with custom scale
        /// </summary>
        /// <param name="scale">The constant scaling factor</param>
        public static GradScaler CreateStatic(float scale);

        /// <summary>
        /// Creates a dynamic loss scaler with conservative settings
        /// </summary>
        public static GradScaler CreateConservative();

        /// <summary>
        /// Creates a dynamic loss scaler with aggressive settings
        /// </summary>
        public static GradScaler CreateAggressive();

        /// <summary>
        /// Creates a loss scaler optimized for FP16 training
        /// </summary>
        public static GradScaler CreateForFP16();

        /// <summary>
        /// Creates a loss scaler optimized for BF16 training
        /// </summary>
        public static GradScaler CreateForBF16();

        /// <summary>
        /// Creates a loss scaler with custom configuration
        /// </summary>
        /// <param name="config">The dynamic scaler configuration</param>
        public static GradScaler CreateFromConfig(DynamicScalerConfig config);
    }
}
```

### 3. GradScalerContext Class

**File:** `src/MLFramework/Amp/GradScalerContext.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Context manager for using GradScaler with automatic cleanup
    /// </summary>
    public class GradScalerContext : IDisposable
    {
        private readonly GradScaler _scaler;
        private readonly Tensor _scaledLoss;
        private bool _stepped;

        /// <summary>
        /// Gets the scaled loss
        /// </summary>
        public Tensor ScaledLoss => _scaledLoss;

        /// <summary>
        /// Creates a new GradScalerContext
        /// </summary>
        /// <param name="scaler">The GradScaler to use</param>
        /// <param name="loss">The loss tensor to scale</param>
        public GradScalerContext(GradScaler scaler, Tensor loss);

        /// <summary>
        /// Performs optimizer step with unscaling and updates scale
        /// </summary>
        /// <param name="optimizer">The optimizer to step</param>
        /// <returns>True if step was performed, false if skipped</returns>
        public bool Step(IOptimizer optimizer);

        /// <summary>
        /// Performs optimizer step with unscaling (manual update)
        /// </summary>
        /// <param name="optimizer">The optimizer to step</param>
        /// <param name="updateScale">Whether to update the scale factor</param>
        /// <returns>True if step was performed, false if skipped</returns>
        public bool Step(IOptimizer optimizer, bool updateScale);

        /// <summary>
        /// Disposes the context and cleans up resources
        /// </summary>
        public void Dispose();
    }
}
```

## Implementation Details

### Step Method Logic

```csharp
public bool Step(IOptimizer optimizer, bool optimizerStep = true)
{
    var grads = optimizer.GetGradients();
    bool hasOverflow = _scaler.CheckOverflow(grads);

    // Update scale (for dynamic scalers)
    _scaler.UpdateScale(hasOverflow);

    if (hasOverflow)
    {
        // Skip optimizer step on overflow
        return false;
    }

    // Unscale gradients
    var unscaledGrads = _scaler.UnscaleGradients(grads);

    // Set unscaled gradients back to optimizer
    optimizer.SetGradients(unscaledGrads);

    // Perform optimizer step if requested
    if (optimizerStep)
    {
        optimizer.Step();
    }

    return true;
}
```

### Context Manager Pattern

```csharp
using (var context = new GradScalerContext(scaler, loss))
{
    loss.Backward();
    context.Step(optimizer);
}
```

### Factory Defaults

- **CreateDefault()**: Dynamic scaler with moderate settings
- **CreateStatic()**: Static scaler with scale = 65536
- **CreateConservative()**: Dynamic scaler, growthInterval = 5000
- **CreateAggressive()**: Dynamic scaler, growthInterval = 1000
- **CreateForFP16()**: Dynamic scaler, growthInterval = 2000
- **CreateForBF16()**: Static scaler, scale = 1.0 (BF16 needs less scaling)

## Usage Examples

### Basic Usage
```csharp
// Create GradScaler
var scaler = new GradScaler(); // Dynamic scaler by default

// Training loop
for (int epoch = 0; epoch < epochs; epoch++)
{
    var loss = model.Forward(inputs);
    loss = scaler.Scale(loss);
    loss.Backward();
    scaler.Step(optimizer);
}
```

### With Context Manager
```csharp
using (var ctx = new GradScalerContext(scaler, loss))
{
    loss.Backward();
    bool stepped = ctx.Step(optimizer);
    if (!stepped)
    {
        Console.WriteLine("Skipped step due to overflow");
    }
}
```

### Custom Configuration
```csharp
var scaler = GradScalerFactory.CreateForFP16();
// or
var scaler = GradScalerFactory.CreateFromConfig(
    new DynamicScalerConfig
    {
        InitialScale = 32768.0f,
        GrowthFactor = 2.0f,
        BackoffFactor = 0.5f,
        GrowthInterval = 3000
    });
```

### Manual Control
```csharp
var scaler = new GradScaler();
var loss = model.Forward(inputs);
loss = scaler.Scale(loss);
loss.Backward();

var grads = model.GetGradients();
var unscaledGrads = scaler.Unscale(grads);

if (!scaler.CheckOverflow(grads))
{
    optimizer.Step(unscaledGrads);
    scaler.Update();
}
```

## Dependencies
- MLFramework.Core (Tensor, IOptimizer)
- MLFramework.Amp (ILossScaler, StaticLossScaler, DynamicLossScaler, DynamicScalerConfig)
- System (IDisposable)

## Testing Requirements
- Test default constructor creates dynamic scaler
- Test static scaler constructor works correctly
- Test custom scaler constructor works correctly
- Test Step() method with overflow detection
- Test Step() with optimizerStep = false
- Test context manager pattern
- Test factory methods
- Test Enable/Disable methods
- Test GetScaleTensor() method
- Test GetStats() returns correct values

## Success Criteria
- [ ] All constructors work correctly
- [ ] Step() handles overflow properly
- [ ] Context manager works as expected
- [ ] Factory methods create correct configurations
- [ ] Enable/Disable methods work correctly
- [ ] GetScaleTensor() returns correct tensor
- [ ] GetStats() returns accurate statistics
- [ ] All unit tests pass
- [ ] Documentation includes usage examples
