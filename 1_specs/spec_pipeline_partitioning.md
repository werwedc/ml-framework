# Spec: Pipeline Layer Partitioning

## Overview
Implement algorithms to partition a model's layers across multiple pipeline stages, supporting both automatic and manual partitioning.

## Class Design

### PartitionMode
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Strategy for partitioning layers across pipeline stages
    /// </summary>
    public enum PartitionMode
    {
        /// <summary>
        /// Automatically partition based on memory and computation cost
        /// </summary>
        Automatic,

        /// <summary>
        /// User specifies which layers belong to each stage
        /// </summary>
        Manual,

        /// <summary>
        /// Evenly distribute layers across stages
        /// </summary>
        Uniform
    }
}
```

### LayerPartitioner
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Partitions a model into pipeline stages
    /// </summary>
    public class LayerPartitioner
    {
        private readonly PartitionMode _mode;
        private readonly int _numStages;
        private readonly Device[] _devices;

        public LayerPartitioner(
            PartitionMode mode,
            int numStages,
            Device[]? devices = null);

        /// <summary>
        /// Partitions a model into pipeline stages
        /// </summary>
        /// <param name="model">The model to partition</param>
        /// <param name="manualPartitions">
        /// For manual mode: List of layer indices for each stage
        /// Example: [[0,1,2], [3,4,5], [6,7,8,9]] for 3 stages
        /// </param>
        /// <returns>List of PipelineStage objects</returns>
        public List<PipelineStage> Partition(
            Module model,
            List<List<int>>? manualPartitions = null);

        /// <summary>
        /// Automatic partitioning based on memory estimation
        /// </summary>
        private List<List<int>> AutomaticPartition(Module model);

        /// <summary>
        /// Uniform partitioning: evenly distribute layers
        /// </summary>
        private List<List<int>> UniformPartition(Module model);

        /// <summary>
        /// Creates a SequentialModule from a list of layer indices
        /// </summary>
        private Module CreateStageModule(Module model, List<int> layerIndices);

        /// <summary>
        /// Estimates memory usage for a set of layers
        /// </summary>
        private long EstimateMemoryUsage(Module model, List<int> layerIndices);

        /// <summary>
        /// Estimates computation time for a set of layers
        /// </summary>
        private float EstimateComputationTime(Module model, List<int> layerIndices);
    }
}
```

### PartitionResult
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Result of model partitioning
    /// </summary>
    public class PartitionResult
    {
        /// <summary>
        /// List of pipeline stages
        /// </summary>
        public List<PipelineStage> Stages { get; }

        /// <summary>
        /// Layer indices assigned to each stage
        /// </summary>
        public List<List<int>> StageLayerIndices { get; }

        /// <summary>
        /// Estimated memory per stage
        /// </summary>
        public long[] MemoryPerStage { get; }

        /// <summary>
        /// Estimated computation per stage
        /// </summary>
        public float[] ComputationPerStage { get; }

        /// <summary>
        /// Load balance metric (lower is better, 1.0 is perfectly balanced)
        /// </summary>
        public float LoadBalance { get; }
    }
}
```

## Implementation Requirements

### Automatic Partitioning Algorithm
1. **Memory-based Greedy Partitioning**
   - Start with all layers in the first stage
   - Iteratively move layers from overloaded stages to underloaded stages
   - Goal: minimize max memory across stages
   - Constraint: layers must remain in order (no reordering)

2. **Computation-aware Partitioning**
   - Combine memory and FLOPs estimates
   - Weight: 70% memory, 30% computation (configurable)
   - Target: minimize max memory * computation per stage

3. **Fallback to Uniform**
   - If automatic partitioning fails, fall back to uniform
   - Log a warning about the fallback

### Manual Partitioning
1. Validate that `manualPartitions` has exactly `_numStages` entries
2. Validate that all layer indices are unique and in valid range
3. Validate that layers are in order (no gaps or reordering)
4. Create stages exactly as specified

### Uniform Partitioning
1. Divide total layer count evenly across stages
2. Distribute remainder (if any) across first (numLayers % numStages) stages
3. Ensure all layers are assigned

## Testing Requirements

1. **Unit Tests**
   - Test uniform partitioning with even layers
   - Test uniform partitioning with uneven layers
   - Test manual partitioning with valid input
   - Test manual partitioning throws on invalid input (gaps, reordering)
   - Test automatic partitioning with small model
   - Test partitioning preserves layer order

2. **Integration Tests**
   - Test partitioning actual neural network (e.g., ResNet-like)
   - Test each stage has correct layers
   - Test memory estimation is reasonable
   - Test load balancing metric is calculated correctly

3. **Edge Cases**
   - Test partitioning single-layer model
   - Test partitioning with more stages than layers
   - Test empty model (should throw)

## Files to Create
- `src/Pipeline/PartitionMode.cs`
- `src/Pipeline/LayerPartitioner.cs`
- `src/Pipeline/PartitionResult.cs`
- `tests/Pipeline/LayerPartitionerTests.cs`

## Dependencies
- `PipelineStage` from spec_pipeline_stage_core
- Existing `Module`, `SequentialModule`, `Device` classes
- No new external dependencies

## Time Estimate
45-60 minutes for implementation and tests
