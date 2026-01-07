# Spec: DistributedDataParallel Module

## Overview
Implement the high-level DistributedDataParallel (DDP) module that wraps a model and automatically handles gradient synchronization during backward pass.

## Requirements
- Wrap any Module and automatically sync gradients
- Support gradient bucketing for performance
- Handle gradient hooks for automatic synchronization
- Support `findUnusedParameters` mode for flexible models
- Broadcast initial weights from rank 0 to all ranks

## Classes

### 1. DistributedDataParallel Class
```csharp
public class DistributedDataParallel : Module
{
    private readonly Module _module;
    private readonly ProcessGroup _processGroup;
    private readonly bool _findUnusedParameters;
    private readonly GradientBucketManager _bucketManager;
    private readonly HashSet<string> _unusedParameters;

    public DistributedDataParallel(
        Module module,
        ProcessGroup processGroup,
        bool findUnusedParameters = false)
        : base($"DDP({module.Name})")
    {
        _module = module;
        _processGroup = processGroup;
        _findUnusedParameters = findUnusedParameters;
        _bucketManager = new GradientBucketManager(processGroup, module.GetParameters());
        _unusedParameters = new HashSet<string>();

        // Register gradient hooks for automatic synchronization
        RegisterGradientHooks();

        // Broadcast initial weights from rank 0
        BroadcastParameters();
    }

    /// <summary>
    /// Forward pass delegates to the wrapped module.
    /// </summary>
    public override Tensor Forward(Tensor input)
    {
        _unusedParameters.Clear();
        return _module.Forward(input);
    }

    /// <summary>
    /// Get the wrapped module.
    /// </summary>
    public Module Module => _module;

    /// <summary>
    /// Get the process group.
    /// </summary>
    public ProcessGroup ProcessGroup => _processGroup;

    /// <summary>
    /// Get unused parameters from the last forward pass.
    /// Only populated when findUnusedParameters = true.
    /// </summary>
    public IReadOnlyCollection<string> UnusedParameters => _unusedParameters;

    /// <summary>
    /// Reduce all gradients asynchronously.
    /// Called automatically by gradient hooks, but can be called manually.
    /// </summary>
    public Task ReduceGradientsAsync(ReduceOp op = ReduceOp.Sum)
    {
        return _bucketManager.ReduceAllAsync(op);
    }

    /// <summary>
    /// Synchronize parameters across all ranks.
    /// Broadcast from rank 0 to all other ranks.
    /// </summary>
    public void BroadcastParameters()
    {
        if (_processGroup.Rank == 0)
        {
            // Rank 0 broadcasts its parameters to all other ranks
            foreach (var param in _module.GetParameters())
            {
                _processGroup.Broadcast(param, root: 0);
            }
        }
        else
        {
            // Other ranks receive broadcast from rank 0
            foreach (var param in _module.GetParameters())
            {
                _processGroup.Broadcast(param, root: 0);
            }
        }
    }

    private void RegisterGradientHooks();

    private void OnGradientComputed(Tensor gradient, Parameter parameter);

    private async Task ReduceGradientsOnBackward();

    public override IEnumerable<Parameter> GetParameters();

    public override IEnumerable<Parameter> GetNamedParameters();
}
```

### 2. GradientSynchronizationHook Class (Internal)
```csharp
/// <summary>
/// Hook that is attached to parameters to trigger gradient reduction.
/// </summary>
internal class GradientSynchronizationHook
{
    private readonly DistributedDataParallel _ddp;
    private readonly Parameter _parameter;
    private readonly GradientBucketManager _bucketManager;

    public GradientSynchronizationHook(
        DistributedDataParallel ddp,
        Parameter parameter,
        GradientBucketManager bucketManager)
    {
        _ddp = ddp;
        _parameter = parameter;
        _bucketManager = bucketManager;
    }

    /// <summary>
    /// Called when a gradient is computed during backward pass.
    /// </summary>
    public Tensor OnGradient(Tensor gradient)
    {
        // Mark parameter as used
        if (_ddp._findUnusedParameters)
        {
            _ddp._unusedParameters.Remove(_parameter.Name);
        }

        // Trigger bucket reduction
        var bucketIndex = _bucketManager.GetBucketIndex(gradient);
        _bucketManager.ReduceBucketAsync(bucketIndex);

        return gradient;
    }
}
```

## Implementation Details

### Gradient Hook Registration

Each parameter's gradient gets a hook that:
1. Marks the parameter as "used" (for `findUnusedParameters` mode)
2. Triggers asynchronous reduction of its bucket

**Hook Registration**:
```csharp
private void RegisterGradientHooks()
{
    foreach (var param in _module.GetParameters())
    {
        if (_findUnusedParameters)
        {
            _unusedParameters.Add(param.Name);
        }

        var hook = new GradientSynchronizationHook(this, param, _bucketManager);
        param.RegisterGradHook(hook.OnGradient);
    }
}
```

### Backward Pass Flow

**During Backward Pass**:
1. Gradients are computed in reverse topological order
2. Each gradient triggers its bucket to reduce asynchronously
3. Buckets reduce in parallel as gradients become available

**After Backward Pass**:
1. Wait for all bucket reductions to complete
2. Copy reduced gradients back to parameters
3. Optimizer can now step with synchronized gradients

### findUnusedParameters Mode

**Purpose**: Some models have conditional paths where not all parameters are used for every input.

**Behavior**:
- Track which parameters receive gradients during forward/backward
- After backward pass, parameters without gradients remain unchanged
- Slower due to tracking overhead, but more flexible

**Implementation**:
- Initialize all parameters as "unused" before each forward pass
- Clear the set when gradient hook is called
- After backward, `_unusedParameters` contains parameters that didn't get gradients

### Parameter Broadcasting

**When**: During DistributedDataParallel construction

**Purpose**: Ensure all ranks start with identical model weights

**Process**:
1. Rank 0 broadcasts each parameter tensor to all other ranks
2. Other ranks receive and copy the broadcast values
3. Uses the process group's Broadcast primitive

### Gradient Synchronization

**Strategy**: Use GradientBucketManager for efficient reduction

**Flow**:
1. Hooks trigger bucket reduction as gradients are computed
2. Buckets reduce asynchronously using AllReduce
3. After backward pass, wait for all buckets and copy back

**Timing**:
- Reduction happens during backward pass (overlapping with computation)
- Synchronization completes before optimizer step

## Usage Example

```csharp
// Initialize process group
var processGroup = ProcessGroup.Init(BackendType.NCCL);

// Create model and wrap with DDP
var model = new LargeModel();
model = new DistributedDataParallel(model, processGroup, findUnusedParameters: false);

// Create optimizer (works with DDP-wrapped model)
var optimizer = new Adam(model.GetParameters(), lr: 0.001);

// Training loop (same as single-GPU)
for (int epoch = 0; epoch < numEpochs; epoch++)
{
    foreach (var batch in dataLoader)
    {
        // Forward
        var output = model.Forward(batch.Data);
        var loss = lossFn(output, batch.Label);

        // Backward (gradients automatically synchronized)
        loss.Backward();

        // Optimizer step (uses synchronized gradients)
        optimizer.Step();
        optimizer.ZeroGrad();
    }
}
```

## Success Criteria
- [ ] Wraps any Module and preserves forward pass behavior
- [ ] Gradients are automatically synchronized during backward pass
- [ ] Parameter broadcasting initializes all ranks identically
- [ ] Works with gradient bucketing
- [ ] findUnusedParameters mode correctly tracks unused parameters
- [ ] Compatible with optimizers (gradients available when needed)

## Dependencies
- spec_communication_backend_interface.md (ProcessGroup, Tensor, ReduceOp)
- spec_process_group.md (ProcessGroup implementation)
- spec_gradient_bucketing.md (GradientBucketManager)
- Existing Module class and autograd engine

## Testing
- Unit tests in spec_ddp_tests.md will verify:
  - Forward pass behavior matches wrapped module
  - Gradients are correctly synchronized
  - Parameter broadcasting works
  - findUnusedParameters tracking is correct
  - Integration with optimizers
  - Edge cases (single GPU, no parameters, etc.)
