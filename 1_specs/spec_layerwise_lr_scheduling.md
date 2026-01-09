# Spec: Layer-Wise Learning Rate Scheduling

## Overview
Implement support for different learning rates for frozen vs. unfrozen parameters during fine-tuning.

## Requirements

### 1. Extension Methods for Optimizer
Add methods to support layer-wise learning rates:
- `SetLayerWiseLrs(this Optimizer optimizer, Dictionary<string, float> layerLrs)`: Set specific LR for each layer
- `SetLayerWiseLrs(this Optimizer optimizer, float frozen, float unfrozen)`: Set LRs for frozen vs. unfrozen
- `SetLayerWiseLrs(this Optimizer optimizer, float[] lrSchedule, string[] layerNames)`: Set LR schedule per layer
- `GetParameterGroups(this Optimizer optimizer)`: Return current parameter groups with their LRs

### 2. ParameterGroup Class
Define groups of parameters with shared optimizer settings:
- `Parameters`: List of parameters in group
- `LearningRate`: LR for this group
- `WeightDecay`: Weight decay for this group
- `Momentum`: Momentum value (if applicable)
- Name/Identifier: Group name for debugging

### 3. ParameterGroupBuilder
Fluent API to build parameter groups:
- `AddFrozenParameters(Model model)`: Add all frozen parameters
- `AddUnfrozenParameters(Model model)`: Add all unfrozen parameters
- `AddLayer(string layerName, Model model)`: Add specific layer
- `AddLayersByPattern(string pattern, Model model)`: Add layers matching regex
- `WithLearningRate(float lr)`: Set LR for current group
- `WithWeightDecay(float wd)`: Set weight decay for current group
- `Build()`: Return list of parameter groups

### 4. LayerWiseLrScheduler
Scheduler for layer-wise learning rate updates:
- `SetMultiplier(string layerPattern, float multiplier)`: Multiply base LR for matching layers
- `SetMultiplierByLayerIndex(int layerIndex, float multiplier)`: Set multiplier for specific layer
- `GetCurrentLrs(float baseLr)`: Get current LRs for all parameter groups
- Support different schedules per layer (linear, cosine, step)

### 5. Common LR Schedules for Fine-Tuning
Pre-configured schedules:
- `DiscriminativeLrSchedule(float baseLr, float[] multipliers)`: Decreasing LRs from early to late layers
- `TriangularLrSchedule(float minLr, float maxLr)`: Triangular schedule for cyclical learning rates
- `WarmsUpCosineSchedule(float baseLr, float warmupLr, int warmupSteps)`: Warmup followed by cosine annealing

### 6. Integration with Optimizer
- Optimizer should support multiple parameter groups
- Each group can have separate hyperparameters
- Step() applies group-specific updates

### 7. Unit Tests
Test cases for:
- Set layer-wise LRs by frozen/unfrozen status
- Set layer-wise LRs by specific layer names
- Parameter group builder creates correct groups
- LR schedules apply correct multipliers
- Optimizer step respects group settings
- Edge cases (all parameters frozen, single group, empty model)

## Files to Create
- `src/ModelZoo/TransferLearning/LayerWiseLrExtensions.cs`
- `src/ModelZoo/TransferLearning/ParameterGroup.cs`
- `src/ModelZoo/TransferLearning/ParameterGroupBuilder.cs`
- `src/ModelZoo/TransferLearning/LayerWiseLrScheduler.cs`
- `tests/ModelZooTests/TransferLearning/LayerWiseLrTests.cs`

## Dependencies
- Existing Optimizer class
- Existing Model and Parameter classes

## Success Criteria
- Can set different LRs for frozen and unfrozen layers
- Parameter groups are correctly applied during optimization
- Common fine-tuning schedules work correctly
- Test coverage > 90%
