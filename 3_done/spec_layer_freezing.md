# Spec: Layer Freezing and Unfreezing

## Overview
Implement utilities to freeze and unfreeze model parameters for transfer learning workflows.

## Requirements

### 1. Extension Methods for Model
Add extension methods to the Model class:
- `Freeze(this Model model)`: Freeze all parameters in the model
- `Unfreeze(this Model model)`: Unfreeze all parameters in the model
- `Freeze(this Model model, int exceptLastN)`: Freeze all except last N layers
- `FreezeByName(this Model model, params string[] layerNames)`: Freeze specific layers by name
- `UnfreezeByName(this Model model, params string[] layerNames)`: Unfreeze specific layers by name
- `FreezeByNamePattern(this Model model, string regexPattern)`: Freeze layers matching regex pattern
- `UnfreezeByNamePattern(this Model model, string regexPattern)`: Unfreeze layers matching regex pattern

### 2. Parameter State Management
Track parameter state:
- Add `RequiresGrad` property to Parameter class
- Maintain list of frozen/unfrozen parameters per model
- Support querying frozen state: `model.GetFrozenLayers()`, `model.GetUnfrozenLayers()`

### 3. LayerSelectionHelper Class
Helper for selecting layers:
- `SelectByName(string name)`: Select layer by exact name
- `SelectByPattern(string pattern)`: Select layers matching regex
- `SelectByIndex(int index)`: Select layer by position
- `SelectByRange(int start, int end)`: Select range of layers
- `SelectByType(Type layerType)`: Select all layers of given type

### 4. GradualUnfreezingScheduler
Implement progressive unfreezing for fine-tuning:
- `Schedule(double[] unfreezeThresholds)`: Define schedule (e.g., [0.1, 0.3, 0.5, 0.7])
- `UpdateUnfreezing(double epochProgress)`: Unfreeze layers based on training progress
- Example: At 10% of epochs, unfreeze last 1 layer; at 30%, unfreeze last 2 layers; etc.

### 5. Frozen Parameter Visualization
- `PrintFrozenState(this Model model)`: Console output showing frozen/unfrozen status
- `GetFrozenStateSummary(this Model model)`: Return summary with counts

### 6. Unit Tests
Test cases for:
- Freeze and unfreeze entire model
- Freeze all except last N layers
- Freeze/unfreeze by name
- Freeze/unfreeze by regex pattern
- Gradual unfreezing schedule
- Query frozen state
- Layer selection helpers
- Edge cases (empty model, single layer, N > total layers)

## Files to Create
- `src/ModelZoo/TransferLearning/FreezeExtensions.cs`
- `src/ModelZoo/TransferLearning/LayerSelectionHelper.cs`
- `src/ModelZoo/TransferLearning/GradualUnfreezingScheduler.cs`
- `tests/ModelZooTests/TransferLearning/FreezingTests.cs`

## Dependencies
- Existing Model and Parameter classes
- System.Text.RegularExpressions (for pattern matching)

## Success Criteria
- Can freeze/unfreeze any subset of layers
- Gradual unfreezing follows schedule correctly
- Parameter state is preserved across training
- Test coverage > 90%
