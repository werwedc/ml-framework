# Spec: Gradient Accumulation System

## Overview
Implement gradient accumulation support for batch processing, enabling efficient training with large batches that don't fit in memory.

## Files to Create
- `src/MLFramework/Autograd/GradientAccumulator.cs`
- `src/MLFramework/Autograd/AccumulationContext.cs`
- `tests/MLFramework.Tests/Autograd/GradientAccumulationTests.cs`

## API Design

### Class: AccumulationContext
```csharp
public class AccumulationContext : IDisposable
{
    public int AccumulationSteps { get; private set; }
    public int TargetSteps { get; }
    public bool IsReady { get; }

    public AccumulationContext(int targetSteps);
    public void RegisterTensor(Tensor tensor);
    public void Step();
    public void Reset();
    public void Dispose();
}
```

### Class: GradientAccumulator
```csharp
public class GradientAccumulator
{
    public int AccumulationCount { get; }
    public bool Enabled { get; set; }

    public GradientAccumulator(int accumulationCount);
    public void EnableAccumulation(Tensor[] parameters);
    public void DisableAccumulation();
    public void Step();
    public void ApplyGradients(Action<Tensor> optimizerStep);
    public void ResetGradients();
}
```

### Extension Methods for Tensor
```csharp
public static class TensorAccumulationExtensions
{
    public static void EnableGradAccumulation(this Tensor tensor);
    public static void DisableGradAccumulation(this Tensor tensor);
    public static bool HasGradAccumulation(this Tensor tensor);
}
```

## Requirements

### Core Functionality
1. **Gradient Accumulation**
   - Accumulate gradients across multiple forward/backward passes
   - Scale gradients by 1/accumulation_steps automatically
   - Only apply optimizer step after target steps reached

2. **Context Management**
   - Track accumulation state for each tensor
   - Register tensors that require accumulation
   - Reset accumulation after optimizer step

3. **Automatic Scaling**
   - Scale accumulated gradients by 1/target_steps
   - Equivalent to averaging gradients across mini-batches
   - Maintain numerical stability

4. **Control Flow**
   - Check if accumulation is complete (`IsReady`)
   - Manual step control (`Step()`)
   - Reset to start new accumulation cycle

## Implementation Notes

### Accumulation Strategy
- Add gradients to existing gradient (not replace)
- Count steps automatically
- Apply scaling at optimizer step time
- Optional manual scaling for custom optimizers

### Memory Management
- Reuse gradient tensors across accumulation steps
- Avoid allocating new tensors for each step
- Dispose accumulation context when done

### Integration with Training Loop
```csharp
var accumulator = new GradientAccumulator(4);
accumulator.EnableAccumulation(parameters);

for (int epoch = 0; epoch < epochs; epoch++)
{
    for (int batch = 0; batch < numBatches; batch++)
    {
        // Forward pass
        var output = model(inputs[batch]);
        var loss = criterion(output, targets[batch]);

        // Backward pass (accumulates gradients)
        loss.Backward();

        // Check if ready to step optimizer
        if (accumulator.IsReady)
        {
            optimizer.Step();  // Applies scaled gradients
            optimizer.ZeroGrad();
        }
    }
}
```

### Thread Safety
- Thread-safe gradient accumulation
- Support for multi-GPU gradient accumulation
- Lock-free operations where possible

## Testing Requirements

### Unit Tests
1. Enable accumulation on tensor → verify state
2. Accumulate gradients over 3 steps → verify sum
3. Verify automatic scaling (1/steps)
4. Test `IsReady` flag at different steps
5. Test gradient reset after optimizer step
6. Test disable accumulation → clears state
7. Test accumulation with multiple tensors

### Integration Tests
1. Train linear regression with batch size 4, accumulation 2 → equivalent to batch size 8
2. Compare loss trajectory with/without accumulation → should match
3. Test gradient accumulation with momentum optimizer
4. Test gradient accumulation with Adam optimizer
5. Test memory usage with large accumulation steps

## Dependencies
- Tensor gradient tracking
- Backward pass implementation
- Memory management system

## Success Criteria
- Accurate gradient accumulation
- Automatic scaling works correctly
- Memory-efficient (no extra allocations per step)
- Integrates seamlessly with optimizers
- Thread-safe for multi-threaded scenarios
