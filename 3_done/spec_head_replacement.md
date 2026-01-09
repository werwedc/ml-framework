# Spec: Head Replacement Utilities

## Overview
Implement utilities to replace final layers of models for transfer learning tasks.

## Requirements

### 1. Extension Methods for Model
Add extension methods for head manipulation:
- `RemoveLastLayer(this Model model)`: Remove the final layer
- `RemoveLastNLayers(this Model model, int n)`: Remove the final N layers
- `AddHead(this Model model, Module head)`: Add a new module as the head
- `ReplaceHead(this Model model, Module newHead)`: Replace the final layer with new module
- `RemoveHead(this Model model, string headName)`: Remove layer by name
- `GetHead(this Model model)`: Get the final layer
- `GetHeads(this Model model)`: Get all final N classification/regression layers

### 2. Common Head Builders
Helper methods to create common heads:
- `LinearHead(int inputDim, int numClasses, bool includeDropout = false, float dropoutRate = 0.5f)`: Create linear classification head
- `MLPHead(int inputDim, int[] hiddenDims, int outputDim)`: Create multi-layer perceptron head
- `ConvHead(int inputChannels, int numClasses, int[] kernelSizes = null)`: Create convolutional head
- `AdaptiveAvgPoolHead(int outputSize, int numClasses)`: Create head with global average pooling
- `AttentionHead(int inputDim, int numClasses, int numHeads = 8)`: Create attention-based head

### 3. HeadAdapter Class
Adapter for managing head replacement:
- `SetNumberOfClasses(int numClasses)`: Dynamically adjust head for different numbers of classes
- `InitializeWeights(WeightInitializationStrategy strategy)`: Initialize new head weights
- `GetInputDim()`: Get expected input dimension for head

### 4. Weight Initialization Strategies
Enum with strategies:
- Xavier (Glorot)
- Kaiming (He)
- Uniform
- Normal
- TruncatedNormal
- Orthogonal

### 5. Head Replacement Validation
Ensure safe replacement:
- Validate input/output dimensions match
- Validate compatible layer types
- Throw descriptive exceptions for invalid operations

### 6. Unit Tests
Test cases for:
- Remove last layer
- Remove last N layers
- Add linear head
- Replace head with different architecture
- Head builder methods create correct structures
- Weight initialization produces valid ranges
- Dimension validation
- Edge cases (single layer model, empty head)

## Files to Create
- `src/ModelZoo/TransferLearning/HeadExtensions.cs`
- `src/ModelZoo/TransferLearning/HeadBuilder.cs`
- `src/ModelZoo/TransferLearning/HeadAdapter.cs`
- `src/ModelZoo/TransferLearning/WeightInitializationStrategy.cs`
- `tests/ModelZooTests/TransferLearning/HeadReplacementTests.cs`

## Dependencies
- Existing Model and Module classes
- Existing initialization utilities (if any)

## Success Criteria
- Can safely replace heads on ResNet, BERT, and other architectures
- New heads are correctly initialized
- Dimension mismatches are caught before runtime
- Test coverage > 90%
