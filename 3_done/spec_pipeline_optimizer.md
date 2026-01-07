# Spec: Pipeline Optimizer Integration

## Overview
Integrate pipeline parallelism with the optimizer system, including gradient synchronization across pipeline stages and parameter updates.

## Class Design

### PipelineOptimizer
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Optimizer wrapper for pipeline parallel training
    /// Handles gradient synchronization and parameter updates across stages
    /// </summary>
    public class PipelineOptimizer : IDisposable
    {
        private readonly IOptimizer _baseOptimizer;
        private readonly List<PipelineStage> _stages;
        private readonly IPipelineCommunicator _communicator;
        private readonly bool _synchronizeGradients;

        /// <summary>
        /// Number of pipeline stages
        /// </summary>
        public int NumStages => _stages.Count;

        /// <summary>
        /// Rank of this stage
        /// </summary>
        public int CurrentRank { get; }

        public PipelineOptimizer(
            IOptimizer baseOptimizer,
            List<PipelineStage> stages,
            IPipelineCommunicator communicator,
            bool synchronizeGradients = true);

        /// <summary>
        /// Perform a single optimization step
        /// </summary>
        public void Step();

        /// <summary>
        /// Zero the gradients for all parameters
        /// </summary>
        public void ZeroGrad();

        /// <summary>
        /// Set gradients for parameters (for testing)
        /// </summary>
        public void SetGradients(List<Tensor> gradients);

        /// <summary>
        /// Synchronize gradients across all pipeline stages
        /// </summary>
        private void SynchronizeGradients();

        /// <summary>
        /// Broadcast updated parameters from stage 0 to all stages
        /// </summary>
        private void BroadcastParameters();

        /// <summary>
        /// Get current learning rate
        /// </summary>
        public float LearningRate => _baseOptimizer.LearningRate;

        /// <summary>
        /// Set learning rate
        /// </summary>
        public void SetLearningRate(float lr);

        /// <summary>
        /// Get optimizer state for inspection
        /// </summary>
        public Dictionary<string, object> GetState();

        /// <summary>
        /// Load optimizer state
        /// </summary>
        public void LoadState(Dictionary<string, object> state);

        public void Dispose();
    }
}
```

### GradientSyncMode
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Strategy for gradient synchronization across pipeline stages
    /// </summary>
    public enum GradientSyncMode
    {
        /// <summary>
        /// No synchronization (each stage has its own copy)
        /// </summary>
        None,

        /// <summary>
        /// Average gradients across all stages (each stage has same model)
        /// </summary>
        Average,

        /// <summary>
        /// Sum gradients across all stages
        /// </summary>
        Sum,

        /// <summary>
        /// Each stage updates only its parameters (model partitioning)
        /// </summary>
        StageWise
    }
}
```

### PipelineOptimizerConfig
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Configuration for pipeline optimizer
    /// </summary>
    public class PipelineOptimizerConfig
    {
        /// <summary>
        /// Gradient synchronization mode
        /// </summary>
        public GradientSyncMode SyncMode { get; set; } = GradientSyncMode.Average;

        /// <summary>
        /// Whether to synchronize gradients before optimizer step
        /// </summary>
        public bool SynchronizeGradients { get; set; } = true;

        /// <summary>
        /// Whether to broadcast parameters after optimizer step
        /// </summary>
        public bool BroadcastParameters { get; set; } = true;

        /// <summary>
        /// Communication timeout in milliseconds
        /// </summary>
        public int CommunicationTimeoutMs { get; set; } = 30000;

        public void Validate();
    }
}
```

## Implementation Requirements

### PipelineOptimizer Constructor
1. Validate that stages list is not empty
2. Validate that base optimizer matches stages
3. Store reference to current rank from communicator

### ZeroGrad
1. Call `ZeroGrad` on all pipeline stages
2. This will zero gradients for all parameters in each stage

### Step
1. **If synchronize gradients:**
   - Collect gradients from all stages
   - Reduce gradients based on `SyncMode` (Average/Sum)
   - Broadcast reduced gradients back to all stages
2. **Call optimizer step:**
   - Run `Step()` on base optimizer for each stage
3. **If broadcast parameters:**
   - Send updated parameters from stage 0 to all other stages
   - Ensure all stages have consistent parameters

### Gradient Synchronization (Average Mode)
1. Each stage sends its gradients to stage 0 (or to all-reduce)
2. Stage 0 averages gradients across all stages
3. Stage 0 broadcasts averaged gradients to all stages
4. All stages set their gradients to the averaged values

### Parameter Broadcasting
1. Stage 0 sends its updated parameters to all other stages
2. Other stages receive and copy the parameters
3. Ensure parameter shapes match
4. Use communicator for actual data transfer

### Gradient Sync Modes
- **None**: Each stage updates independently (fast, inconsistent)
- **Average**: Gradients are averaged across stages (consistent)
- **Sum**: Gradients are summed across stages (for large batch simulation)
- **StageWise**: Each stage updates only its parameters (model partitioning)

### State Management
1. `GetState` returns base optimizer state plus pipeline-specific info
2. `LoadState` restores state and ensures consistency across stages
3. Include learning rate, momentum, and other optimizer hyperparameters

## Testing Requirements

1. **Unit Tests**
   - Test optimizer creation with valid inputs
   - Test ZeroGrad clears all gradients
   - Test Step with synchronization enabled
   - Test Step with synchronization disabled
   - Test gradient averaging produces correct result
   - Test gradient sum produces correct result
   - Test parameter broadcasting updates all stages
   - Test learning rate getter/setter
   - Test state save/load

2. **Integration Tests**
   - Test full training iteration with optimizer
   - Test that all stages have same parameters after training
   - Test with different gradient sync modes
   - Test with actual neural network
   - Test optimizer convergence (compare with single-device)

3. **Edge Cases**
   - Test single-stage pipeline (degenerate case)
   - Test with empty gradient list
   - Test communication timeout handling
   - Test inconsistent parameter shapes (should throw)

## Files to Create
- `src/Pipeline/PipelineOptimizer.cs`
- `src/Pipeline/GradientSyncMode.cs`
- `src/Pipeline/PipelineOptimizerConfig.cs`
- `tests/Pipeline/PipelineOptimizerTests.cs`

## Dependencies
- `PipelineStage` from spec_pipeline_stage_core
- `IPipelineCommunicator` from spec_pipeline_communication
- Existing `IOptimizer` interface
- Existing autograd system

## Time Estimate
45-60 minutes for implementation and tests

## Notes
- This is critical for correct training with pipeline parallelism
- Gradient synchronization ensures consistent updates across stages
- Parameter broadcasting ensures model consistency
- Future: Integrate with ZeRO-style optimizer state sharding
