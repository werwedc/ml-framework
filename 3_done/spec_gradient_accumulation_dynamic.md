# Spec: Gradient Accumulation for Variable Batch Sizes

## Overview
Implement gradient accumulation that supports variable batch sizes across accumulation steps.

## Requirements

### Class: DynamicBatchAccumulator
- Properties:
  - `AccumulatedGradient`: Tensor
  - `AccumulatedBatchSize`: int
  - `CurrentStep`: int
  - `TargetBatchSize`: int
  - `AccumulationCount`: int

- Methods:
  - `Accumulate(Tensor batchGradient, int batchSize)`: void
  - `IsComplete()`: bool
  - `GetAccumulatedGradient()`: Tensor
  - `Reset()`: void
  - `GetProgress()`: double

### Class: GradientScaling
- Methods:
  - `ScaleByBatchSize(Tensor gradient, int batchSize, int referenceSize)`: Tensor
  - `AverageAccumulated(Tensor accumulated, int totalBatchSize, int referenceSize)`: Tensor
  - `NormalizeBatchGradient(Tensor gradient, int batchSize)`: Tensor

### Class: VariableBatchScheduler
- Properties:
  - `BatchSchedule`: List<int> - List of batch sizes per step
  - `CurrentStep`: int
  - `TotalSteps`: int
  - `EffectiveBatchSize`: int

- Methods:
  - `GetCurrentBatchSize()`: int
  - `GetRemainingSteps()`: int
  - `GetEffectiveBatchSize(int windowSize)`: int - Average over window
  - `Advance()`: void

### Class: AccumulationBufferDynamic
- Properties:
  - `Buffer`: Tensor
  - `BufferShape`: SymbolicShape
  - `CurrentSize`: int
  - `MaxSize`: int

- Methods:
  - `Accumulate(Tensor gradient, int startIdx, int count)`: void
  - `Resize(int newSize)`: void
  - `GetSlice(int startIdx, int count)`: Tensor
  - `GetFull()`: Tensor

### Gradient Accumulation Strategies:

1. **Fixed Target Batch**: Accumulate until reaching target batch size
   - Variable batch sizes per step
   - Normalize by actual batch size

2. **Fixed Step Count**: Accumulate for fixed number of steps
   - Actual batch size may vary
   - Average gradients at end

3. **Adaptive Accumulation**: Adjust based on batch size variance
   - More accumulation for smaller batches
   - Less for larger batches

### Class: GradientAccumulationValidator
- Methods:
  - `ValidateAccumulation(List<Tensor> gradients, List<int> batchSizes)`: bool
  - `CheckShapeCompatibility(Tensor grad1, Tensor grad2)`: bool
  - `ValidateAccumulatedShape(Tensor accumulated, List<Tensor> components)`: bool

### Unit Tests
- Test accumulation with constant batch size
- Test accumulation with variable batch sizes
- Test scaling and normalization
- Test fixed target batch strategy
- Test fixed step count strategy
- Test adaptive accumulation
- Test buffer resizing
- Test gradient validation
- Test edge cases (empty accumulation, overflow)

## Implementation Notes
- Normalize gradients by actual batch size to ensure fairness
- Support partial accumulation for efficiency
- Track accumulated batch size accurately
- Handle symbolic shapes in buffer management
- Provide progress tracking for monitoring

## Dependencies
- spec_autograd_dynamic_shapes.md
- spec_symbolic_shape.md

## Success Criteria
- Correctly accumulates gradients with variable batch sizes
- Normalization prevents bias toward larger batches
- Buffer management handles size changes efficiently
- Validation catches shape mismatches
- Progress tracking is accurate
