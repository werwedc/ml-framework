# Spec: Gradient Checkpointing System

## Overview
Implement gradient checkpointing (activation recomputation) to trade computation for memory, enabling training of larger models that don't fit in GPU memory.

## Files to Create
- `src/MLFramework/Autograd/CheckpointManager.cs`
- `src/MLFramework/Autograd/CheckpointScope.cs`
- `src/MLFramework/Autograd/CheckpointNode.cs`
- `tests/MLFramework.Tests/Autograd/CheckpointingTests.cs`

## API Design

### Class: CheckpointScope : IDisposable
```csharp
public class CheckpointScope : IDisposable
{
    public string Name { get; }
    public bool IsEnabled { get; set; }
    public bool UseRecomputation { get; set; }

    public CheckpointScope(string name, bool enabled = true);
    public void Dispose();
}
```

### Class: CheckpointNode
```csharp
public class CheckpointNode : GraphNode
{
    public bool IsCheckpoint { get; set; }
    public List<Tensor> SavedActivations { get; }
    public List<Func<Tensor[]>> RecomputeFunctions { get; }

    public CheckpointNode(Tensor output, OperationContext operation, params GraphNode[] children);
    public void SaveActivations(params Tensor[] activations);
    public void AddRecomputeFunction(Func<Tensor[]> recomputeFn);
    public Tensor[] Recompute();
    public void ClearSavedActivations();
}
```

### Class: CheckpointManager
```csharp
public class CheckpointManager
{
    public int MaxMemoryMB { get; set; }
    public bool AutoCheckpoint { get; set; }
    public float MemoryThreshold { get; set; }

    public CheckpointManager();
    public void RegisterScope(CheckpointScope scope);
    public void UnregisterScope(CheckpointScope scope);
    public bool ShouldCheckpoint(GraphNode node);
    public void SelectiveCheckpointing(List<GraphNode> nodes);
    public void ClearAllCheckpoints();
}
```

### Extension Methods
```csharp
public static class TensorCheckpointExtensions
{
    public static Tensor MarkCheckpoint(this Tensor tensor);
    public static bool IsCheckpoint(this Tensor tensor);
}
```

## Usage Examples

### Manual Checkpointing
```csharp
var input = Tensor.Random(256, 256, requiresGrad: true);

// First block (checkpoint)
Tensor x1;
using (var checkpoint1 = new CheckpointScope("layer1"))
{
    x1 = largeLayer1(input);
}

// Intermediate block (no checkpoint)
var x2 = largeLayer2(x1);
var x3 = largeLayer3(x2);

// Second block (checkpoint)
Tensor x4;
using (var checkpoint2 = new CheckpointScope("layer4"))
{
    x4 = largeLayer4(x3);
}

var output = largeLayer5(x4);
var loss = output.Sum();

// Backward pass will recompute x1 and x4
loss.Backward();
```

### Automatic Checkpointing
```csharp
var manager = new CheckpointManager
{
    AutoCheckpoint = true,
    MaxMemoryMB = 1024,
    MemoryThreshold = 0.8f
};

// Manager automatically selects checkpoints based on memory
var output = model(input);
loss.Backward();
```

### Layer-Level Checkpointing
```csharp
public class CheckpointedLayer
{
    public Tensor Forward(Tensor x)
    {
        using (var scope = new CheckpointScope("layer"))
        {
            return layer.Forward(x);
        }
    }
}
```

## Requirements

### Core Functionality
1. **Checkpoint Marking**
   - Mark specific operations as checkpoints
   - Save intermediate activations at checkpoints
   - Store recompute functions

2. **Selective Activation Storage**
   - Only store activations at checkpoint locations
   - Discard intermediate activations
   - Track memory usage

3. **On-Demand Recomputation**
   - Recompute activations during backward pass
   - Call recompute functions in order
   - Cache recomputed activations for multiple gradient paths

4. **Memory-Computation Tradeoff**
   - Configure memory threshold
   - Auto-select checkpoints based on memory
   - Manual checkpoint override

### Automatic Checkpointing Strategy
1. Analyze memory usage of each operation
2. Select operations that would save most memory
3. Ensure checkpoints are evenly distributed
4. Avoid too frequent recompute (performance degradation)

## Implementation Notes

### Recomputation Logic
- During backward, if node needs activation:
  1. Check if it's a checkpoint → retrieve saved activation
  2. Otherwise → find nearest upstream checkpoint
  3. Recompute from checkpoint to required node

### Memory Management
- Track memory usage of saved activations
- Dispose activations when no longer needed
- Use memory threshold to decide checkpoint frequency

### Graph Integration
- CheckpointNode extends GraphNode
- Backward pass handles recomputation automatically
- No changes needed to user code (except marking checkpoints)

### Performance Considerations
- Recomputation adds computational overhead
- Tradeoff: memory savings vs. compute cost
- Typical: 2-3x compute time, 50-70% memory savings

## Testing Requirements

### Unit Tests
1. Test checkpoint scope creation/disposal
2. Test activation saving and retrieval
3. Test recomputation function execution
4. Test backward pass with checkpoints
5. Test memory savings calculation
6. Test auto-checkpoint selection
7. Test multiple checkpoints in sequence
8. Test nested checkpoints

### Integration Tests
1. Train model with checkpointing → verify memory reduction
2. Compare accuracy with/without checkpointing → should match
3. Test checkpointing with residual networks
4. Test checkpointing with attention layers
5. Measure memory savings on large model
6. Benchmark recompute overhead

## Dependencies
- Computational graph infrastructure
- Backward pass implementation
- Memory management system
- Tensor operations

## Success Criteria
- Reduces memory usage by 50-70% for typical models
- Gradients match non-checkpointed version
- User-friendly API (minimal code changes)
- Automatic checkpointing works reliably
- Reasonable computational overhead (< 3x)
