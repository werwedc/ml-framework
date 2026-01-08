# Spec: Autograd Integration

## Overview
Implement integration with the automatic differentiation (autograd) system to automatically trigger recomputation during the backward pass. This enables seamless checkpointing with minimal user code changes.

## Classes

### Location
`src/MLFramework/Checkpointing/AutogradIntegration.cs`

### Class: CheckpointFunction

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Autograd function that implements checkpointing for backward pass recomputation
/// </summary>
public class CheckpointFunction : AutogradFunction
{
    private readonly string _layerId;
    private readonly Func<Tensor> _forwardFunc;
    private readonly Action<Tensor>? _backwardHook;
    private readonly CheckpointManager _checkpointManager;
    private readonly RecomputationEngine _recomputeEngine;

    /// <summary>
    /// Initializes a new instance of CheckpointFunction
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="forwardFunc">Forward pass function</param>
    /// <param name="backwardHook">Optional backward hook</param>
    /// <param name="checkpointManager">Checkpoint manager instance</param>
    /// <param name="recomputeEngine">Recomputation engine instance</param>
    public CheckpointFunction(
        string layerId,
        Func<Tensor> forwardFunc,
        Action<Tensor>? backwardHook,
        CheckpointManager checkpointManager,
        RecomputationEngine recomputeEngine);

    /// <summary>
    /// Forward pass - executes the function and optionally checkpoints
    /// </summary>
    /// <param name="inputs">Input tensors</param>
    /// <returns>Output tensor</returns>
    protected override Tensor Forward(params Tensor[] inputs);

    /// <summary>
    /// Backward pass - recomputes if needed and computes gradients
    /// </summary>
    /// <param name="gradOutput">Gradient from subsequent layers</param>
    /// <returns>Gradients for input tensors</returns>
    protected override Tensor[] Backward(params Tensor[] gradOutput);
}
```

## Implementation Details

### CheckpointFunction Constructor

```csharp
public CheckpointFunction(
    string layerId,
    Func<Tensor> forwardFunc,
    Action<Tensor>? backwardHook,
    CheckpointManager checkpointManager,
    RecomputationEngine recomputeEngine)
{
    _layerId = layerId ?? throw new ArgumentNullException(nameof(layerId));
    _forwardFunc = forwardFunc ?? throw new ArgumentNullException(nameof(forwardFunc));
    _backwardHook = backwardHook;
    _checkpointManager = checkpointManager ?? throw new ArgumentNullException(nameof(checkpointManager));
    _recomputeEngine = recomputeEngine ?? throw new ArgumentNullException(nameof(recomputeEngine));

    // Register the recompute function
    _recomputeEngine.RegisterRecomputeFunction(layerId, forwardFunc);
}
```

### CheckpointFunction.Forward

```csharp
protected override Tensor Forward(params Tensor[] inputs)
{
    // Execute forward pass
    var output = _forwardFunc();

    // Determine if we should checkpoint (based on strategy)
    var shouldCheckpoint = ShouldCheckpoint(_layerId);

    if (shouldCheckpoint)
    {
        // Save inputs for potential recomputation
        // We need to save them in a context that Backward can access
        SaveInputsForRecomputation(inputs);

        // Register backward hook if provided
        if (_backwardHook != null)
        {
            output.RegisterBackwardHook(_backwardHook);
        }
    }
    else
    {
        // Not checkpointing - this activation will be stored normally
        // No special handling needed
    }

    return output;
}

private bool ShouldCheckpoint(string layerId)
{
    // This will be determined by the checkpoint strategy
    // For now, we'll delegate to a strategy manager
    return false; // Placeholder - will be implemented in strategy spec
}

private void SaveInputsForRecomputation(Tensor[] inputs)
{
    // Store inputs in a way that Backward can access
    // This could be in a thread-local context or a class field
    // Implementation depends on the autograd system's capabilities
}
```

### CheckpointFunction.Backward

```csharp
protected override Tensor[] Backward(params Tensor[] gradOutput)
{
    // Retrieve saved inputs
    var inputs = RetrieveSavedInputs();

    // Recompute the forward pass if needed
    var output = _recomputeEngine.Recompute(_layerId);

    // Compute gradients using the recomputed output
    var gradients = ComputeGradients(inputs, output, gradOutput);

    return gradients;
}

private Tensor[] ComputeGradients(Tensor[] inputs, Tensor output, Tensor[] gradOutput)
{
    // Implement gradient computation
    // This will depend on the specific operation
    // For now, return empty as a placeholder
    return Array.Empty<Tensor>();
}
```

## Checkpoint Decorator

### Class: Checkpoint

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Static class providing convenient methods for checkpointing functions
/// </summary>
public static class Checkpoint
{
    /// <summary>
    /// Checkpoints a function during the forward pass
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="func">Function to checkpoint</param>
    /// <param name="checkpointManager">Checkpoint manager (optional, uses default if null)</param>
    /// <param name="recomputeEngine">Recompute engine (optional, uses default if null)</param>
    /// <returns>Result of the function</returns>
    public static Tensor CheckpointFunction(
        string layerId,
        Func<Tensor> func,
        CheckpointManager? checkpointManager = null,
        RecomputationEngine? recomputeEngine = null)
    {
        var manager = checkpointManager ?? GetDefaultCheckpointManager();
        var engine = recomputeEngine ?? GetDefaultRecomputeEngine();

        var checkpointFunc = new CheckpointFunction(
            layerId,
            func,
            null,
            manager,
            engine);

        return checkpointFunc.Apply();
    }

    /// <summary>
    /// Creates a checkpointed version of a module/layers
    /// </summary>
    /// <typeparam name="T">Type of the module</typeparam>
    /// <param name="module">Module to checkpoint</param>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>Checkpointed module wrapper</returns>
    public static T CheckpointModule<T>(T module, string layerId)
        where T : class
    {
        // Implementation will wrap the module's forward pass with checkpointing
        throw new NotImplementedException();
    }

    private static CheckpointManager GetDefaultCheckpointManager()
    {
        // Return or create a default singleton instance
        throw new NotImplementedException();
    }

    private static RecomputationEngine GetDefaultRecomputeEngine()
    {
        // Return or create a default singleton instance
        throw new NotImplementedException();
    }
}
```

## Backward Hook System

### Class: BackwardHookManager

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Manages backward hooks for checkpointing
/// </summary>
public class BackwardHookManager : IDisposable
{
    /// <summary>
    /// Registers a backward hook for a specific layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="hook">Hook function to call during backward pass</param>
    /// <returns>Handle that can be used to remove the hook</returns>
    public int RegisterHook(string layerId, Action<Tensor> hook);

    /// <summary>
    /// Removes a previously registered hook
    /// </summary>
    /// <param name="handle">Handle returned from RegisterHook</param>
    public void RemoveHook(int handle);

    /// <summary>
    /// Removes all hooks for a specific layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    public void RemoveHooksForLayer(string layerId);

    /// <summary>
    /// Removes all registered hooks
    /// </summary>
    public void ClearAllHooks();

    /// <summary>
    /// Invokes hooks for a specific layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="gradient">Gradient tensor</param>
    public void InvokeHooks(string layerId, Tensor gradient);

    /// <summary>
    /// Disposes the manager and releases resources
    /// </summary>
    public void Dispose();

    private readonly Dictionary<int, (string LayerId, Action<Tensor> Hook)> _hooks;
    private int _nextHandle;
}
```

## Gradient Accumulation Integration

### Class: CheckpointedGradientAccumulator

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Gradient accumulator that works with checkpointed activations
/// </summary>
public class CheckpointedGradientAccumulator : IDisposable
{
    /// <summary>
    /// Initializes a new instance of CheckpointedGradientAccumulator
    /// </summary>
    /// <param name="accumulationSteps">Number of steps to accumulate</param>
    public CheckpointedGradientAccumulator(int accumulationSteps);

    /// <summary>
    /// Accumulates gradients for a batch
    /// </summary>
    /// <param name="gradients">Gradients to accumulate</param>
    public void Accumulate(Dictionary<string, Tensor> gradients);

    /// <summary>
    /// Gets the accumulated gradients and resets the accumulator
    /// </summary>
    /// <returns>Accumulated gradients</returns>
    public Dictionary<string, Tensor> GetAccumulatedGradients();

    /// <summary>
    /// Resets the accumulator without returning gradients
    /// </summary>
    public void Reset();

    /// <summary>
    /// Checks if it's time to apply accumulated gradients
    /// </summary>
    /// <returns>True if accumulation steps reached, false otherwise</returns>
    public bool ShouldApplyGradients();

    /// <summary>
    /// Disposes the accumulator and releases resources
    /// </summary>
    public void Dispose();

    private readonly int _accumulationSteps;
    private int _currentStep;
    private readonly Dictionary<string, Tensor> _accumulatedGradients;
}
```

## Context Management

### Class: CheckpointContext

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Manages checkpointing context for a forward/backward pass
/// </summary>
public class CheckpointContext : IDisposable
{
    /// <summary>
    /// Initializes a new instance of CheckpointContext
    /// </summary>
    /// <param name="config">Checkpoint configuration</param>
    public CheckpointContext(CheckpointConfig config);

    /// <summary>
    /// Enters the checkpoint context (enables checkpointing)
    /// </summary>
    public void Enter();

    /// <summary>
    /// Exits the checkpoint context (disables checkpointing and cleans up)
    /// </summary>
    public void Exit();

    /// <summary>
    /// Gets the current checkpoint configuration
    /// </summary>
    public CheckpointConfig Config { get; }

    /// <summary>
    /// Gets whether checkpointing is currently enabled
    /// </summary>
    public bool IsEnabled { get; private set; }

    /// <summary>
    /// Disposes the context and exits if still active
    /// </summary>
    public void Dispose();

    private CheckpointManager _checkpointManager;
    private RecomputationEngine _recomputeEngine;
    private MemoryTracker _memoryTracker;
}
```

## Testing Requirements

### Unit Tests

1. **CheckpointFunction Tests**
   - [ ] Forward pass executes correctly
   - [ ] Backward pass triggers recomputation
   - [ ] Backward hook is called correctly
   - [ ] Inputs are properly saved for recomputation
   - [ ] Gradients are computed correctly

2. **Checkpoint Decorator Tests**
   - [ ] CheckpointFunction creates correct CheckpointFunction
   - [ ] Default managers are created correctly
   - [ ] Custom managers are used when provided
   - [ ] CheckpointModule wraps module correctly

3. **BackwardHookManager Tests**
   - [ ] Hook registration returns unique handle
   - [ ] Hook removal works correctly
   - [ ] Multiple hooks can be registered for same layer
   - [ ] Hooks are invoked in correct order
   - [ ] ClearAllHooks removes all hooks

4. **CheckpointedGradientAccumulator Tests**
   - [ ] Gradients are accumulated correctly
   - [ ] ShouldApplyGradients returns correct value
   - [ ] GetAccumulatedGradients returns correct gradients
   - [ ] Reset clears accumulated gradients
   - [ ] Accumulation steps are respected

5. **CheckpointContext Tests**
   - [ ] Enter enables checkpointing
   - [ ] Exit disables checkpointing
   - [ ] Context is properly cleaned up on exit
   - [ ] Config is preserved correctly
   - [ ] IsEnabled reflects current state

6. **Integration Tests**
   - [ ] End-to-end forward/backward with checkpointing
   - [ ] Gradients match non-checkpointed version
   - [ ] Memory is reduced with checkpointing
   - [ ] Multiple forward/backward passes work correctly

7. **Edge Cases**
   - [ ] Handle null forward function
   - [ ] Handle exceptions in forward function
   - [ ] Handle exceptions in backward function
   - [ ] Handle multiple contexts simultaneously
   - [ ] Handle nested checkpoint contexts

8. **Thread Safety Tests**
   - [ ] Multiple forward passes can execute concurrently
   - [ ] Contexts are thread-safe
   - [ ] Hook manager is thread-safe

## Implementation Notes

1. **Autograd System Integration**:
   - Must integrate with the existing autograd framework
   - Respect autograd's existing hooks and callbacks
   - Ensure compatibility with custom autograd functions

2. **Performance**:
   - Minimize overhead in forward pass
   - Optimize backward pass to reduce recomputation
   - Use efficient data structures for storing inputs

3. **Correctness**:
   - Ensure gradients are numerically identical
   - Handle edge cases in gradient computation
   - Preserve gradient metadata

4. **Error Handling**:
   - Provide clear error messages
   - Allow recovery from failures
   - Maintain state consistency

## Dependencies on Other Specs

This spec depends on:
- **Checkpoint Manager Core** (spec_1) for CheckpointManager
- **Recomputation Engine** (spec_4) for RecomputationEngine
- **Checkpoint Configuration** (spec_2) for CheckpointConfig

## Estimated Implementation Time
45-60 minutes
