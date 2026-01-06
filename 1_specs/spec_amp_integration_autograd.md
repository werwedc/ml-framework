# Spec: AMP Autograd Integration

## Overview
Integrate Automatic Mixed Precision with the autograd system to ensure gradient computation works correctly with mixed precision tensors.

## Class Specification

### 1. AmpAutogradContext Class

**File:** `src/MLFramework/Amp/Integrations/AmpAutogradContext.cs`

```csharp
namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Context for AMP-aware autograd operations
    /// </summary>
    public class AmpAutogradContext
    {
        /// <summary>
        /// Gets or sets the current AutoCast mode
        /// </summary>
        public AutoCastMode Mode { get; set; }

        /// <summary>
        /// Gets or sets the operation precision registry
        /// </summary>
        public AmpRegistry Registry { get; set; }

        /// <summary>
        /// Gets or sets the loss scaler
        /// </summary>
        public ILossScaler? LossScaler { get; set; }

        /// <summary>
        /// Gets whether gradient unscaling is needed
        /// </summary>
        public bool NeedsGradientUnscaling { get; set; }

        /// <summary>
        /// Creates a new AmpAutogradContext
        /// </summary>
        public AmpAutogradContext(
            AutoCastMode mode = AutoCastMode.Bf16,
            AmpRegistry? registry = null,
            ILossScaler? lossScaler = null,
            bool needsGradientUnscaling = false);
    }
}
```

### 2. AmpAutogradFunction Class

**File:** `src/MLFramework/Amp/Integrations/AmpAutogradFunction.cs`

```csharp
namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Base class for AMP-aware autograd functions
    /// </summary>
    public abstract class AmpAutogradFunction
    {
        /// <summary>
        /// Gets the operation precision registry
        /// </summary>
        protected AmpRegistry Registry { get; }

        /// <summary>
        /// Gets the AutoCast mode
        /// </summary>
        protected AutoCastMode Mode { get; }

        /// <summary>
        /// Creates a new AmpAutogradFunction
        /// </summary>
        protected AmpAutogradFunction(AmpRegistry? registry = null);

        /// <summary>
        /// Forward pass with automatic precision casting
        /// </summary>
        /// <param name="inputs">Input tensors</param>
        /// <param name="operationType">The type of operation</param>
        /// <returns>Output tensors</returns>
        public Tensor[] Forward(Tensor[] inputs, Type operationType);

        /// <summary>
        /// Forward pass with manual precision specification
        /// </summary>
        /// <param name="inputs">Input tensors</param>
        /// <param name="forwardDtype">The dtype for forward pass</param>
        /// <param name="backwardDtype">The dtype for backward pass</param>
        /// <returns>Output tensors</returns>
        public abstract Tensor[] ForwardManual(
            Tensor[] inputs,
            DataType forwardDtype,
            DataType backwardDtype);

        /// <summary>
        /// Backward pass with automatic precision handling
        /// </summary>
        /// <param name="gradOutputs">Gradient outputs</param>
        /// <param name="operationType">The type of operation</param>
        /// <returns>Gradient inputs</returns>
        public Tensor[] Backward(Tensor[] gradOutputs, Type operationType);

        /// <summary>
        /// Backward pass with manual precision specification
        /// </summary>
        /// <param name="gradOutputs">Gradient outputs</param>
        /// <param name="backwardDtype">The dtype for backward pass</param>
        /// <returns>Gradient inputs</returns>
        public abstract Tensor[] BackwardManual(
            Tensor[] gradOutputs,
            DataType backwardDtype);

        /// <summary>
        /// Gets the forward dtype for an operation
        /// </summary>
        protected DataType GetForwardDtype(Type operationType, DataType inputDtype);

        /// <summary>
        /// Gets the backward dtype for an operation
        /// </summary>
        protected DataType GetBackwardDtype(Type operationType, DataType inputDtype);
    }
}
```

### 3. AmpTensor Extension Methods

**File:** `src/MLFramework/Amp/Integrations/AmpTensorExtensions.cs`

```csharp
namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Extension methods for Tensor with AMP support
    /// </summary>
    public static class AmpTensorExtensions
    {
        /// <summary>
        /// Backward pass with automatic AMP handling
        /// </summary>
        /// <param name="tensor">The tensor to compute gradients for</param>
        /// <param name="lossScaler">The loss scaler (optional)</param>
        public static void BackwardAmp(this Tensor tensor, ILossScaler? lossScaler = null);

        /// <summary>
        /// Backward pass with gradient retention and AMP handling
        /// </summary>
        /// <param name="tensor">The tensor to compute gradients for</param>
        /// <param name="retainGraph">Whether to retain the computation graph</param>
        /// <param name="lossScaler">The loss scaler (optional)</param>
        public static void BackwardAmp(
            this Tensor tensor,
            bool retainGraph,
            ILossScaler? lossScaler = null);

        /// <summary>
        /// Gets the gradients with AMP-aware dtype handling
        /// </summary>
        /// <param name="tensor">The tensor to get gradients from</param>
        /// <param name="lossScaler">The loss scaler (optional)</param>
        /// <returns>Gradient tensor with correct dtype</returns>
        public static Tensor GradAmp(this Tensor tensor, ILossScaler? lossScaler = null);

        /// <summary>
        /// Checks if a tensor requires AMP-aware backward pass
        /// </summary>
        /// <param name="tensor">The tensor to check</param>
        /// <returns>True if AMP-aware backward is needed, false otherwise</returns>
        public static bool NeedsAmpBackward(this Tensor tensor);
    }
}
```

### 4. AmpAutogradHelper Class

**File:** `src/MLFramework/Amp/Integrations/AmpAutogradHelper.cs`

```csharp
namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Helper methods for AMP autograd integration
    /// </summary>
    public static class AmpAutogradHelper
    {
        /// <summary>
        /// Prepares gradients for optimizer step in AMP mode
        /// </summary>
        /// <param name="gradients">The gradients to prepare</param>
        /// <param name="lossScaler">The loss scaler</param>
        /// <param name="checkOverflow">Whether to check for overflow</param>
        /// <returns>True if gradients are valid, false if overflow detected</returns>
        public static bool PrepareGradientsForOptimizer(
            Dictionary<string, Tensor> gradients,
            ILossScaler lossScaler,
            bool checkOverflow = true);

        /// <summary>
        /// Converts gradients to the correct dtype for optimizer
        /// </summary>
        /// <param name="gradients">The gradients to convert</param>
        /// <param name="targetDtype">The target dtype (usually Float32)</param>
        /// <returns>Converted gradients</returns>
        public static Dictionary<string, Tensor> ConvertGradientsDtype(
            Dictionary<string, Tensor> gradients,
            DataType targetDtype);

        /// <summary>
        /// Ensures gradient dtype compatibility with parameters
        /// </summary>
        /// <param name="parameters">The parameters</param>
        /// <param name="gradients">The gradients</param>
        /// <returns>True if compatible, false otherwise</returns>
        public static bool EnsureGradientCompatibility(
            Dictionary<string, Tensor> parameters,
            Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Creates an AMP-aware computation graph
        /// </summary>
        /// <param name="tensor">The root tensor</param>
        /// <param name="mode">The AutoCast mode</param>
        /// <param name="registry">The operation registry</param>
        /// <returns>An AMP-aware computation graph</returns>
        public static ComputationGraph CreateAmpGraph(
            Tensor tensor,
            AutoCastMode mode = AutoCastMode.Bf16,
            AmpRegistry? registry = null);

        /// <summary>
        /// Runs an AMP-aware backward pass
        /// </summary>
        /// <param name="graph">The computation graph</param>
        /// <param name="lossScaler">The loss scaler</param>
        /// <returns>Gradients with correct dtype</returns>
        public static Dictionary<string, Tensor> RunAmpBackward(
            ComputationGraph graph,
            ILossScaler lossScaler);
    }
}
```

### 5. AmpCustomFunction Class

**File:** `src/MLFramework/Amp/Integrations/AmpCustomFunction.cs`

```csharp
namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Base class for custom AMP-aware autograd functions
    /// </summary>
    /// <typeparam name="TContext">The context type for the function</typeparam>
    public abstract class AmpCustomFunction<TContext> : AmpAutogradFunction
        where TContext : AmpAutogradContext, new()
    {
        /// <summary>
        /// Creates a new AmpCustomFunction
        /// </summary>
        protected AmpCustomFunction(AmpRegistry? registry = null)
            : base(registry) { }

        /// <summary>
        /// Creates the context for the forward/backward pass
        /// </summary>
        /// <param name="inputs">Input tensors</param>
        /// <returns>The context</returns>
        protected abstract TContext CreateContext(Tensor[] inputs);

        /// <summary>
        /// Forward pass implementation
        /// </summary>
        /// <param name="ctx">The context</param>
        /// <param name="inputs">Input tensors</param>
        /// <returns>Output tensors</returns>
        protected abstract Tensor[] ForwardImpl(TContext ctx, Tensor[] inputs);

        /// <summary>
        /// Backward pass implementation
        /// </summary>
        /// <param name="ctx">The context</param>
        /// <param name="gradOutputs">Gradient outputs</param>
        /// <returns>Gradient inputs</returns>
        protected abstract Tensor[] BackwardImpl(TContext ctx, Tensor[] gradOutputs);

        /// <summary>
        /// Forward pass with manual precision specification
        /// </summary>
        public override Tensor[] ForwardManual(
            Tensor[] inputs,
            DataType forwardDtype,
            DataType backwardDtype)
        {
            var ctx = CreateContext(inputs);
            ctx.Mode = MapDataTypeToMode(forwardDtype);
            return ForwardImpl(ctx, inputs);
        }

        /// <summary>
        /// Backward pass with manual precision specification
        /// </summary>
        public override Tensor[] BackwardManual(
            Tensor[] gradOutputs,
            DataType backwardDtype)
        {
            var ctx = new TContext();
            ctx.Mode = MapDataTypeToMode(backwardDtype);
            return BackwardImpl(ctx, gradOutputs);
        }

        private AutoCastMode MapDataTypeToMode(DataType dtype)
        {
            return dtype switch
            {
                DataType.Float16 => AutoCastMode.Fp16,
                DataType.BFloat16 => AutoCastMode.Bf16,
                _ => AutoCastMode.None
            };
        }
    }
}
```

## Implementation Details

### Autograd Integration Points

1. **Forward Pass Integration**
   - Hook into tensor operations to apply AutoCast
   - Cast tensors to appropriate precision based on registry
   - Store original dtypes for backward pass

2. **Backward Pass Integration**
   - Check if AMP context is active
   - Apply gradient dtype conversions
   - Handle loss scaling if present

3. **Gradient Flow**
   - Ensure gradients are in correct dtype for optimizer
   - Handle mixed precision gradient accumulation
   - Support gradient clipping with AMP

### Gradient Preparation

```csharp
public static bool PrepareGradientsForOptimizer(
    Dictionary<string, Tensor> gradients,
    ILossScaler lossScaler,
    bool checkOverflow = true)
{
    if (checkOverflow && lossScaler != null)
    {
        bool hasOverflow = lossScaler.CheckOverflow(gradients);
        if (hasOverflow)
        {
            return false;
        }
    }

    // Unscale gradients
    if (lossScaler != null)
    {
        gradients = lossScaler.UnscaleGradients(gradients);
    }

    // Convert gradients to Float32 for optimizer
    gradients = ConvertGradientsDtype(gradients, DataType.Float32);

    return true;
}
```

### Dtype Conversion

```csharp
public static Dictionary<string, Tensor> ConvertGradientsDtype(
    Dictionary<string, Tensor> gradients,
    DataType targetDtype)
{
    var result = new Dictionary<string, Tensor>();

    foreach (var (name, grad) in gradients)
    {
        if (grad.Dtype == targetDtype)
        {
            result[name] = grad;
        }
        else
        {
            result[name] = grad.Cast(targetDtype);
        }
    }

    return result;
}
```

## Usage Examples

### Basic AMP Autograd
```csharp
// Create AMP context
var registry = new AmpRegistry(AmpConfig.CreateBf16());
var scaler = new DynamicLossScaler();

// Forward pass
using (var autocast = new AutoCast(AutoCastMode.Bf16, registry))
{
    var output = model.Forward(inputs);
    var loss = criterion(output, targets);

    // Backward pass with AMP
    loss.BackwardAmp(scaler);

    // Prepare gradients for optimizer
    var grads = model.GetGradients();
    if (AmpAutogradHelper.PrepareGradientsForOptimizer(grads, scaler))
    {
        optimizer.Step();
    }
}
```

### Custom AMP Function
```csharp
public class MyCustomFunction : AmpCustomFunction<MyAmpContext>
{
    protected override MyAmpContext CreateContext(Tensor[] inputs)
    {
        return new MyAmpContext
        {
            Mode = AutoCastMode.Bf16,
            Registry = Registry
        };
    }

    protected override Tensor[] ForwardImpl(MyAmpContext ctx, Tensor[] inputs)
    {
        // Custom forward logic with AMP awareness
        var x = AutoCastContext.Cast(inputs[0], typeof(MyCustomOp));
        // ... operation ...
        return new[] { output };
    }

    protected override Tensor[] BackwardImpl(MyAmpContext ctx, Tensor[] gradOutputs)
    {
        // Custom backward logic with AMP awareness
        // ... backward operation ...
        return new[] { gradInput };
    }
}
```

### Gradient Compatibility Check
```csharp
var parameters = model.GetParameters();
var gradients = model.GetGradients();

if (!AmpAutogradHelper.EnsureGradientCompatibility(parameters, gradients))
{
    throw new InvalidOperationException("Gradient dtype mismatch detected");
}
```

## Dependencies
- MLFramework.Core (Tensor, DataType, ComputationGraph)
- MLFramework.Amp (AutoCast, AutoCastMode, AmpRegistry, AmpConfig, ILossScaler)
- MLFramework.Autograd (backward pass integration)
- System.Collections.Generic (Dictionary)

## Testing Requirements
- Test forward pass with automatic casting
- Test backward pass with gradient dtype conversion
- Test gradient preparation with loss scaling
- Test custom AMP function
- Test gradient compatibility checking
- Test overflow detection during backward pass
- Test mixed precision gradient accumulation
- Test with various AutoCast modes

## Success Criteria
- [ ] Forward pass applies correct dtype casting
- [ ] Backward pass produces gradients in correct dtype
- [ ] Gradient preparation works correctly
- [ ] Custom AMP functions work as expected
- [ ] Overflow detection is accurate
- [ ] Gradient compatibility is ensured
- [ ] Performance overhead < 5% of autograd time
- [ ] All unit tests pass
- [ ] Documentation includes usage examples
