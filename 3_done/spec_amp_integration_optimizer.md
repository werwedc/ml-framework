# Spec: AMP Optimizer Integration

## Overview
Integrate Automatic Mixed Precision with the optimizer system to ensure parameter updates work correctly with mixed precision gradients.

## Class Specification

### 1. AmpOptimizerWrapper Class

**File:** `src/MLFramework/Amp/Integrations/AmpOptimizerWrapper.cs`

```csharp
namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Wrapper for optimizers that handles AMP-specific operations
    /// </summary>
    public class AmpOptimizerWrapper : IOptimizer
    {
        private readonly IOptimizer _optimizer;
        private readonly GradScaler _scaler;
        private readonly DataType _parameterDtype;
        private readonly DataType _gradientDtype;

        /// <summary>
        /// Gets the underlying optimizer
        /// </summary>
        public IOptimizer Optimizer => _optimizer;

        /// <summary>
        /// Gets the GradScaler
        /// </summary>
        public GradScaler Scaler => _scaler;

        /// <summary>
        /// Gets the parameter dtype
        /// </summary>
        public DataType ParameterDtype => _parameterDtype;

        /// <summary>
        /// Gets the gradient dtype
        /// </summary>
        public DataType GradientDtype => _gradientDtype;

        /// <summary>
        /// Creates a new AmpOptimizerWrapper
        /// </summary>
        /// <param name="optimizer">The underlying optimizer</param>
        /// <param name="scaler">The GradScaler</param>
        /// <param name="parameterDtype">The parameter dtype (default: Float16/BFloat16)</param>
        /// <param name="gradientDtype">The gradient dtype (default: Float32)</param>
        public AmpOptimizerWrapper(
            IOptimizer optimizer,
            GradScaler scaler,
            DataType? parameterDtype = null,
            DataType? gradientDtype = null);

        /// <summary>
        /// Performs an optimizer step with AMP handling
        /// </summary>
        /// <param name="gradients">The gradients to use for the update</param>
        /// <returns>True if step was performed, false if skipped</returns>
        public bool Step(Dictionary<string, Tensor>? gradients = null);

        /// <summary>
        /// Performs an optimizer step with explicit gradient dictionary
        /// </summary>
        /// <param name="gradients">The gradients to use for the update</param>
        /// <param name="checkOverflow">Whether to check for overflow</param>
        /// <param name="updateScale">Whether to update the loss scale</param>
        /// <returns>True if step was performed, false if skipped</returns>
        public bool Step(
            Dictionary<string, Tensor> gradients,
            bool checkOverflow = true,
            bool updateScale = true);

        /// <summary>
        /// Zeroes the gradients
        /// </summary>
        public void ZeroGrad();

        /// <summary>
        /// Gets the gradients from the model
        /// </summary>
        /// <returns>Dictionary of parameter names to gradient tensors</returns>
        public Dictionary<string, Tensor> GetGradients();

        /// <summary>
        /// Sets the gradients
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        public void SetGradients(Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Gets the parameters
        /// </summary>
        /// <returns>Dictionary of parameter names to parameter tensors</returns>
        public Dictionary<string, Tensor> GetParameters();

        /// <summary>
        /// Sets the learning rate
        /// </summary>
        /// <param name="lr">The new learning rate</param>
        public void SetLearningRate(float lr);

        /// <summary>
        /// Gets the learning rate
        /// </summary>
        /// <returns>The current learning rate</returns>
        public float GetLearningRate();

        /// <summary>
        /// Loads optimizer state
        /// </summary>
        /// <param name="state">The state to load</param>
        public void LoadState(object state);

        /// <summary>
        /// Gets optimizer state
        /// </summary>
        /// <returns>The optimizer state</returns>
        public object GetState();
    }
}
```

### 2. AmpOptimizerHelper Class

**File:** `src/MLFramework/Amp/Integrations/AmpOptimizerHelper.cs`

```csharp
namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Helper methods for AMP optimizer integration
    /// </summary>
    public static class AmpOptimizerHelper
    {
        /// <summary>
        /// Wraps an optimizer with AMP handling
        /// </summary>
        /// <param name="optimizer">The optimizer to wrap</param>
        /// <param name="scaler">The GradScaler</param>
        /// <param name="parameterDtype">The parameter dtype (optional)</param>
        /// <param name="gradientDtype">The gradient dtype (optional)</param>
        /// <returns>An AMP-wrapped optimizer</returns>
        public static AmpOptimizerWrapper WrapOptimizer(
            IOptimizer optimizer,
            GradScaler scaler,
            DataType? parameterDtype = null,
            DataType? gradientDtype = null);

        /// <summary>
        /// Creates an AMP-wrapped SGD optimizer
        /// </summary>
        /// <param name="parameters">The parameters to optimize</param>
        /// <param name="lr">The learning rate</param>
        /// <param name="scaler">The GradScaler</param>
        /// <param name="momentum">The momentum factor (optional)</param>
        /// <param name="dampening">The dampening factor (optional)</param>
        /// <param name="weightDecay">The weight decay (optional)</param>
        /// <param name="nesterov">Whether to use Nesterov momentum (optional)</param>
        /// <returns>An AMP-wrapped SGD optimizer</returns>
        public static AmpOptimizerWrapper CreateSgd(
            Dictionary<string, Tensor> parameters,
            float lr,
            GradScaler scaler,
            float momentum = 0.0f,
            float dampening = 0.0f,
            float weightDecay = 0.0f,
            bool nesterov = false);

        /// <summary>
        /// Creates an AMP-wrapped Adam optimizer
        /// </summary>
        /// <param name="parameters">The parameters to optimize</param>
        /// <param name="lr">The learning rate</param>
        /// <param name="scaler">The GradScaler</param>
        /// <param name="beta1">The beta1 parameter (optional)</param>
        /// <param name="beta2">The beta2 parameter (optional)</param>
        /// <param name="eps">The epsilon term (optional)</param>
        /// <param name="weightDecay">The weight decay (optional)</param>
        /// <param name="amsgrad">Whether to use AMSGrad variant (optional)</param>
        /// <returns>An AMP-wrapped Adam optimizer</returns>
        public static AmpOptimizerWrapper CreateAdam(
            Dictionary<string, Tensor> parameters,
            float lr,
            GradScaler scaler,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float eps = 1e-8f,
            float weightDecay = 0.0f,
            bool amsgrad = false);

        /// <summary>
        /// Creates an AMP-wrapped AdamW optimizer
        /// </summary>
        /// <param name="parameters">The parameters to optimize</param>
        /// <param name="lr">The learning rate</param>
        /// <param name="scaler">The GradScaler</param>
        /// <param name="beta1">The beta1 parameter (optional)</param>
        /// <param name="beta2">The beta2 parameter (optional)</param>
        /// <param name="eps">The epsilon term (optional)</param>
        /// <param name="weightDecay">The weight decay (optional)</param>
        /// <param name="amsgrad">Whether to use AMSGrad variant (optional)</param>
        /// <returns>An AMP-wrapped AdamW optimizer</returns>
        public static AmpOptimizerWrapper CreateAdamW(
            Dictionary<string, Tensor> parameters,
            float lr,
            GradScaler scaler,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float eps = 1e-8f,
            float weightDecay = 0.01f,
            bool amsgrad = false);

        /// <summary>
        /// Creates an AMP-wrapped RMSprop optimizer
        /// </summary>
        /// <param name="parameters">The parameters to optimize</param>
        /// <param name="lr">The learning rate</param>
        /// <param name="scaler">The GradScaler</param>
        /// <param name="alpha">The smoothing constant (optional)</param>
        /// <param name="eps">The epsilon term (optional)</param>
        /// <param name="weightDecay">The weight decay (optional)</param>
        /// <param name="momentum">The momentum factor (optional)</param>
        /// <param name="centered">Whether to use centered RMSprop (optional)</param>
        /// <returns>An AMP-wrapped RMSprop optimizer</returns>
        public static AmpOptimizerWrapper CreateRmsprop(
            Dictionary<string, Tensor> parameters,
            float lr,
            GradScaler scaler,
            float alpha = 0.99f,
            float eps = 1e-8f,
            float weightDecay = 0.0f,
            float momentum = 0.0f,
            bool centered = false);

        /// <summary>
        /// Checks if optimizer parameters are compatible with AMP
        /// </summary>
        /// <param name="parameters">The parameters to check</param>
        /// <param name="targetDtype">The target dtype for parameters</param>
        /// <returns>True if compatible, false otherwise</returns>
        public static bool CheckParameterCompatibility(
            Dictionary<string, Tensor> parameters,
            DataType targetDtype);

        /// <summary>
        /// Converts optimizer parameters to target dtype
        /// </summary>
        /// <param name="parameters">The parameters to convert</param>
        /// <param name="targetDtype">The target dtype</param>
        /// <returns>Converted parameters</returns>
        public static Dictionary<string, Tensor> ConvertParametersDtype(
            Dictionary<string, Tensor> parameters,
            DataType targetDtype);
    }
}
```

### 3. AmpOptimizerState Class

**File:** `src/MLFramework/Amp/Integrations/AmpOptimizerState.cs`

```csharp
namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// State information for AMP-aware optimizers
    /// </summary>
    public class AmpOptimizerState
    {
        /// <summary>
        /// Gets the underlying optimizer state
        /// </summary>
        public object OptimizerState { get; }

        /// <summary>
        /// Gets the GradScaler state
        /// </summary>
        public object ScalerState { get; }

        /// <summary>
        /// Gets the parameter dtype
        /// </summary>
        public DataType ParameterDtype { get; }

        /// <summary>
        /// Gets the gradient dtype
        /// </summary>
        public DataType GradientDtype { get; }

        /// <summary>
        /// Creates a new AmpOptimizerState
        /// </summary>
        public AmpOptimizerState(
            object optimizerState,
            object scalerState,
            DataType parameterDtype,
            DataType gradientDtype);

        /// <summary>
        /// Creates a default AmpOptimizerState
        /// </summary>
        public static AmpOptimizerState CreateDefault(DataType parameterDtype);
    }
}
```

### 4. AmpOptimizerExtension Methods

**File:** `src/MLFramework/Amp/Integrations/AmpOptimizerExtensions.cs`

```csharp
namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Extension methods for IOptimizer with AMP support
    /// </summary>
    public static class AmpOptimizerExtensions
    {
        /// <summary>
        /// Performs an optimizer step with automatic AMP handling
        /// </summary>
        /// <param name="optimizer">The optimizer</param>
        /// <param name="scaler">The GradScaler</param>
        /// <param name="gradients">The gradients (optional)</param>
        /// <returns>True if step was performed, false if skipped</returns>
        public static bool StepAmp(
            this IOptimizer optimizer,
            GradScaler scaler,
            Dictionary<string, Tensor>? gradients = null);

        /// <summary>
        /// Performs an optimizer step with AMP and checks for overflow
        /// </summary>
        /// <param name="optimizer">The optimizer</param>
        /// <param name="scaler">The GradScaler</param>
        /// <param name="gradients">The gradients</param>
        /// <param name="checkOverflow">Whether to check for overflow</param>
        /// <param name="updateScale">Whether to update the loss scale</param>
        /// <returns>True if step was performed, false if skipped</returns>
        public static bool StepAmp(
            this IOptimizer optimizer,
            GradScaler scaler,
            Dictionary<string, Tensor> gradients,
            bool checkOverflow,
            bool updateScale);

        /// <summary>
        /// Gets gradients with AMP-aware dtype handling
        /// </summary>
        /// <param name="optimizer">The optimizer</param>
        /// <param name="scaler">The GradScaler</param>
        /// <returns>Gradients with correct dtype</returns>
        public static Dictionary<string, Tensor> GetGradientsAmp(
            this IOptimizer optimizer,
            GradScaler scaler);

        /// <summary>
        /// Sets gradients with AMP-aware dtype handling
        /// </summary>
        /// <param name="optimizer">The optimizer</param>
        /// <param name="gradients">The gradients</param>
        /// <param name="targetDtype">The target dtype for gradients</param>
        public static void SetGradientsAmp(
            this IOptimizer optimizer,
            Dictionary<string, Tensor> gradients,
            DataType targetDtype);
    }
}
```

## Implementation Details

### Optimizer Step Logic

```csharp
public bool Step(Dictionary<string, Tensor>? gradients = null)
{
    // Get gradients if not provided
    if (gradients == null)
    {
        gradients = _optimizer.GetGradients();
    }

    // Check for overflow
    bool hasOverflow = _scaler.CheckOverflow(gradients);

    // Update scale
    _scaler.UpdateScale(hasOverflow);

    if (hasOverflow)
    {
        // Skip optimizer step on overflow
        return false;
    }

    // Unscale gradients
    var unscaledGrads = _scaler.Unscale(gradients);

    // Convert gradients to target dtype
    var convertedGrads = AmpAutogradHelper.ConvertGradientsDtype(
        unscaledGrads,
        _gradientDtype);

    // Set gradients and step
    _optimizer.SetGradients(convertedGrads);
    _optimizer.Step();

    return true;
}
```

### Parameter Dtype Conversion

```csharp
public static Dictionary<string, Tensor> ConvertParametersDtype(
    Dictionary<string, Tensor> parameters,
    DataType targetDtype)
{
    var result = new Dictionary<string, Tensor>();

    foreach (var (name, param) in parameters)
    {
        if (param.Dtype == targetDtype)
        {
            result[name] = param;
        }
        else
        {
            result[name] = param.Cast(targetDtype);
        }
    }

    return result;
}
```

### Extension Method Implementation

```csharp
public static bool StepAmp(
    this IOptimizer optimizer,
    GradScaler scaler,
    Dictionary<string, Tensor> gradients,
    bool checkOverflow,
    bool updateScale)
{
    if (checkOverflow)
    {
        bool hasOverflow = scaler.CheckOverflow(gradients);
        if (updateScale)
        {
            scaler.UpdateScale(hasOverflow);
        }

        if (hasOverflow)
        {
            return false;
        }
    }

    var unscaledGrads = scaler.Unscale(gradients);
    optimizer.SetGradients(unscaledGrads);
    optimizer.Step();

    return true;
}
```

## Usage Examples

### Basic AMP Optimizer
```csharp
// Create GradScaler
var scaler = new GradScaler();

// Create optimizer
var parameters = model.GetParameters();
var optimizer = new Adam(parameters, lr: 0.001f);

// Wrap optimizer with AMP
var ampOptimizer = AmpOptimizerHelper.WrapOptimizer(optimizer, scaler);

// Training loop
for (int epoch = 0; epoch < epochs; epoch++)
{
    var loss = model.Forward(inputs);
    loss = scaler.Scale(loss);
    loss.Backward();

    ampOptimizer.Step();
    ampOptimizer.ZeroGrad();
}
```

### Using Extension Methods
```csharp
var scaler = new GradScaler();
var optimizer = new Adam(model.GetParameters(), lr: 0.001f);

// Training loop
for (int epoch = 0; epoch < epochs; epoch++)
{
    var loss = model.Forward(inputs);
    loss = scaler.Scale(loss);
    loss.Backward();

    // Use extension method
    optimizer.StepAmp(scaler);
    optimizer.ZeroGrad();
}
```

### Creating AMP Optimizer Directly
```csharp
var scaler = new GradScaler();
var parameters = model.GetParameters();

// Create AMP-wrapped Adam optimizer
var optimizer = AmpOptimizerHelper.CreateAdam(parameters, lr: 0.001f, scaler);

// Training loop
for (int epoch = 0; epoch < epochs; epoch++)
{
    var loss = model.Forward(inputs);
    loss = scaler.Scale(loss);
    loss.Backward();

    optimizer.Step();
    optimizer.ZeroGrad();
}
```

### Parameter Dtype Conversion
```csharp
var parameters = model.GetParameters();

// Convert parameters to BF16 for AMP training
var bf16Params = AmpOptimizerHelper.ConvertParametersDtype(
    parameters,
    DataType.BFloat16);

model.SetParameters(bf16Params);
```

## Dependencies
- MLFramework.Core (Tensor, DataType, IOptimizer)
- MLFramework.Optimizers (SGD, Adam, AdamW, RMSprop)
- MLFramework.Amp (GradScaler, DynamicScalerStats)
- MLFramework.Amp.Integrations (AmpAutogradHelper)
- System.Collections.Generic (Dictionary)

## Testing Requirements
- Test AmpOptimizerWrapper.Step() with overflow detection
- Test AmpOptimizerWrapper.Step() without overflow
- Test parameter dtype conversion
- Test gradient dtype conversion
- Test optimizer state save/load
- Test extension methods
- Test helper methods for creating AMP optimizers
- Test parameter compatibility checking
- Test with various optimizer types (SGD, Adam, AdamW, RMSprop)

## Success Criteria
- [ ] Optimizer step handles overflow correctly
- [ ] Gradient dtype conversion works accurately
- [ ] Parameter dtype conversion works accurately
- [ ] Optimizer state save/load preserves AMP information
- [ ] Extension methods work as expected
- [ ] Helper methods create correct AMP optimizers
- [ ] Parameter compatibility checking is accurate
- [ ] Performance overhead < 5% of optimizer step time
- [ ] All unit tests pass
- [ ] Documentation includes usage examples
