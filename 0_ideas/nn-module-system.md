# Neural Network Module System

## Overview
Create a composable module system for building neural networks with automatic forward/backward pass management, similar to PyTorch's nn.Module.

## Problem
Currently, users need to manually construct computation graphs and manage backward passes. Building even simple networks requires verbose, error-prone tensor manipulation code.

## Feature Requirements

### Core Module Base Class
```csharp
public abstract class Module
{
    public abstract Tensor Forward(Tensor input);
    public virtual void ZeroGrad();
    public virtual void UpdateParameters(IOptimizer optimizer);
    public IEnumerable<Parameter> Parameters();
}
```

### Built-in Layer Modules
1. **Linear Layer**
   - Weight matrix and bias initialization
   - Configurable input/output dimensions
   - Weight initialization options (Xavier, He, Random)

2. **Activation Layers**
   - ReLU, Sigmoid, Tanh, LeakyReLU
   - Automatic gradient computation for activation functions

3. **Sequential Module**
   - Chain multiple modules together
   - Forward pass propagates through chain
   - Parameter collection from all child modules

### Module Composition
- Support for nested modules (modules containing modules)
- Automatic parameter discovery and registration
- Named child modules for model introspection

### Parameter Management
- `Parameter` class wrapping tensors with gradient tracking
- Automatic grad zeroing before backward pass
- Weight normalization and regularization hooks

### Usage Example
```csharp
var model = new Sequential(
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 10)
);

var output = model.Forward(input);
var loss = CrossEntropy(output, target);
loss.Backward();
model.UpdateParameters(optimizer);
```

## Technical Considerations
- Module lifecycle management (training/eval modes)
- Batch normalization and dropout integration points
- Module state serialization/deserialization
- Parameter sharing across modules

## Value
- Reduces boilerplate by 80% for common network architectures
- Provides clean, PyTorch-like API familiar to ML practitioners
- Foundation for complex model architectures (CNNs, RNNs, Transformers)
