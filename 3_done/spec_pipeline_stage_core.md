# Spec: Pipeline Stage Core

## Overview
Define the core data structures for pipeline parallelism, including the PipelineStage class that wraps a module with pipeline metadata.

## Class Design

### PipelineStage
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Represents a single stage in a pipeline parallel training setup.
    /// Wraps a module with metadata about its position in the pipeline.
    /// </summary>
    public class PipelineStage : Module
    {
        /// <summary>
        /// Rank of this stage in the pipeline (0 to TotalStages-1)
        /// </summary>
        public int Rank { get; }

        /// <summary>
        /// Total number of pipeline stages
        /// </summary>
        public int TotalStages { get; }

        /// <summary>
        /// Device this stage executes on
        /// </summary>
        public Device Device { get; }

        /// <summary>
        /// The module containing the layers for this stage
        /// </summary>
        public Module Module { get; }

        /// <summary>
        /// Whether this is the first stage (receives input data)
        /// </summary>
        public bool IsFirstStage => Rank == 0;

        /// <summary>
        /// Whether this is the last stage (produces output)
        /// </summary>
        public bool IsLastStage => Rank == TotalStages - 1;

        /// <summary>
        /// Creates a new pipeline stage
        /// </summary>
        public PipelineStage(
            Module module,
            int rank,
            int totalStages,
            Device device);
    }
}
```

### PipelineConfig
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Configuration for pipeline parallelism setup
    /// </summary>
    public class PipelineConfig
    {
        /// <summary>
        /// Number of pipeline stages (must be <= number of available devices)
        /// </summary>
        public int NumStages { get; set; }

        /// <summary>
        /// Number of micro-batches to split each batch into
        /// </summary>
        public int MicroBatches { get; set; } = 4;

        /// <summary>
        /// Devices to use for each stage (must be length NumStages)
        /// If null, uses first NumStages available devices
        /// </summary>
        public Device[]? Devices { get; set; }

        /// <summary>
        /// Validate configuration
        /// </summary>
        public void Validate();
    }
}
```

## Implementation Requirements

### Constructor Requirements
1. **PipelineStage Constructor**
   - Validate that `rank` is in range [0, totalStages)
   - Validate that `totalStages` > 0
   - Move the provided `module` to the specified `device`
   - Store all parameters and cache them in this stage

2. **PipelineConfig Validation**
   - Assert `NumStages` > 0
   - Assert `MicroBatches` > 0
   - If `Devices` is provided, validate length equals `NumStages`
   - Validate all devices in `Devices` array are available

### Module Overrides
1. Override `Forward()` to delegate to the inner module
2. Override all other `Module` methods to pass through to the wrapped module
3. Ensure parameter tracking works correctly

## Testing Requirements

1. **Unit Tests**
   - Test constructor with valid inputs
   - Test constructor throws on invalid rank
   - Test constructor throws on invalid totalStages
   - Test `IsFirstStage` and `IsLastStage` properties
   - Test module is moved to correct device
   - Test parameter tracking works through wrapper

2. **Integration Tests**
   - Test PipelineStage with actual neural network module
   - Test forward pass through single stage
   - Test backward pass through single stage

## Files to Create
- `src/Pipeline/PipelineStage.cs`
- `src/Pipeline/PipelineConfig.cs`
- `tests/Pipeline/PipelineStageTests.cs`

## Dependencies
- Existing `Module` class
- Existing `Device` class
- No new external dependencies

## Time Estimate
30-45 minutes for implementation and basic tests
