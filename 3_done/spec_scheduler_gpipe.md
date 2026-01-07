# Spec: GPipe Scheduler

## Overview
Implement the 1F1B (One-Forward-One-Backward) scheduling strategy, also known as GPipe-style pipeline parallelism. This is the classic and simplest pipeline scheduling approach.

## Class Design

### SchedulingStrategy
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Pipeline scheduling strategy
    /// </summary>
    public enum SchedulingStrategy
    {
        /// <summary>
        /// Classic GPipe: fill -> steady state -> drain
        /// </summary>
        GPipe,

        /// <summary>
        /// Interleaved 1F1B (PipeDream-Flush) - to be implemented later
        /// </summary>
        Interleaved1F1B
    }
}
```

### GPipeScheduler
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// GPipe-style pipeline scheduler (1F1B scheduling)
    /// </summary>
    public class GPipeScheduler : IDisposable
    {
        private readonly List<PipelineStage> _stages;
        private readonly IPipelineCommunicator _communicator;
        private readonly int _microBatches;
        private readonly List<Tensor> _activations;
        private readonly List<Tensor> _gradients;

        /// <summary>
        /// Number of pipeline stages
        /// </summary>
        public int NumStages => _stages.Count;

        /// <summary>
        /// Number of micro-batches
        /// </summary>
        public int MicroBatches => _microBatches;

        public GPipeScheduler(
            List<PipelineStage> stages,
            IPipelineCommunicator communicator,
            int microBatches);

        /// <summary>
        /// Execute forward pass through the pipeline
        /// </summary>
        /// <returns>Output tensors for each micro-batch</returns>
        public List<Tensor> Forward(Tensor batch);

        /// <summary>
        /// Execute backward pass through the pipeline
        /// </summary>
        /// <returns>Accumulated gradients for all parameters</returns>
        public List<Tensor> Backward(List<Tensor> losses);

        /// <summary>
        /// Execute a single training iteration (forward + backward)
        /// </summary>
        public List<Tensor> TrainIteration(Tensor batch, Func<Tensor, Tensor> lossFn);

        /// <summary>
        /// Get pipeline statistics (bubble time, utilization, etc.)
        /// </summary>
        public PipelineStats GetStats();

        /// <summary>
        /// Reset scheduler state
        /// </summary>
        public void Reset();

        public void Dispose();
    }
}
```

### PipelineStats
```csharp
namespace MLFramework.Pipeline
{
    /// <summary>
    /// Statistics about pipeline execution
    /// </summary>
    public class PipelineStats
    {
        /// <summary>
        /// Total time for forward pass (ms)
        /// </summary>
        public float ForwardTime { get; }

        /// <summary>
        /// Total time for backward pass (ms)
        /// </summary>
        public float BackwardTime { get; }

        /// <summary>
        /// Bubble time (idle time) during forward pass (ms)
        /// </summary>
        public float ForwardBubbleTime { get; }

        /// <summary>
        /// Bubble time (idle time) during backward pass (ms)
        /// </summary>
        public float BackwardBubbleTime { get; }

        /// <summary>
        /// Device utilization (0.0 to 1.0)
        /// </summary>
        public float Utilization => 1.0f - (ForwardBubbleTime + BackwardBubbleTime) / (ForwardTime + BackwardTime);

        public PipelineStats(float forwardTime, float backwardTime, float forwardBubbleTime, float backwardBubbleTime);
    }
}
```

## Implementation Requirements

### GPipe Scheduling Algorithm

The GPipe scheduler has three phases:

#### 1. Fill Phase (Forward)
- Process micro-batches 0, 1, 2, ..., numStages-2
- Each stage i processes micro-batch i and sends to stage i+1
- Stages are idle until first micro-batch arrives

#### 2. Steady State
- Process remaining micro-batches (numStages-1 to microBatches-1)
- Each stage continuously processes one micro-batch per step
- Perfect pipeline utilization

#### 3. Drain Phase (Backward)
- Process backward passes in reverse order
- Each stage i processes micro-batch i, i+1, ..., microBatches-1
- Stages become idle after last micro-batch passes through

### Forward Pass Implementation
1. Split batch into micro-batches
2. For each micro-batch index from 0 to microBatches-1:
   - If not last stage: receive input from previous stage (or use batch data for stage 0)
   - Execute forward pass
   - Store activation for backward pass (if needed)
   - If not last stage: send output to next stage
   - If last stage: store output
3. Return list of outputs from last stage

### Backward Pass Implementation
1. For each micro-batch index from microBatches-1 down to 0:
   - If not first stage: receive gradient from next stage
   - Compute gradients for this stage
   - If not first stage: send gradient to previous stage
   - Accumulate gradients for parameters
2. Return accumulated gradients

### Timing and Statistics
- Use `Stopwatch` to measure timing
- Track idle time (time spent waiting for communication)
- Calculate bubble time and utilization

### Communication Protocol
- Forward: Stage i sends to stage i+1
- Backward: Stage i sends gradients to stage i-1
- Use async operations for non-blocking communication

## Testing Requirements

1. **Unit Tests**
   - Test forward pass with 2-stage pipeline
   - Test forward pass with 4-stage pipeline
   - Test backward pass with 2-stage pipeline
   - Test forward and backward integration
   - Test with different numbers of micro-batches
   - Test that outputs match single-device execution (data consistency)

2. **Integration Tests**
   - Test full training iteration with simple model
   - Test gradient correctness (compare with single-device baseline)
   - Test timing statistics are reasonable
   - Test scheduler reset works correctly

3. **Edge Cases**
   - Test single-stage pipeline (should degenerate to normal execution)
   - Test with microBatches < numStages
   - Test with microBatches = numStages
   - Test with microBatches >> numStages

## Files to Create
- `src/Pipeline/SchedulingStrategy.cs`
- `src/Pipeline/GPipeScheduler.cs`
- `src/Pipeline/PipelineStats.cs`
- `tests/Pipeline/GPipeSchedulerTests.cs`

## Dependencies
- `PipelineStage` from spec_pipeline_stage_core
- `IPipelineCommunicator` from spec_pipeline_communication
- `MicroBatchManager` from spec_microbatch_manager
- Existing `Tensor`, `Module`, autograd system

## Time Estimate
45-60 minutes for implementation and tests

## Notes
- This is the simplest scheduling strategy (GPipe/1F1B)
- Future work: Interleaved 1F1B for better utilization
- Focus on correctness first, optimization later
- Ensure backward pass gradients are accumulated correctly
