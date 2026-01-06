# Spec: Autograd Function Base Class

## Overview
Implement a base class and infrastructure for user-defined custom autograd functions, enabling manual implementation of forward and backward passes for specialized operations.

## Files to Create
- `src/MLFramework/Autograd/AutogradFunction.cs`
- `src/MLFramework/Autograd/FunctionRegistry.cs`
- `tests/MLFramework.Tests/Autograd/AutogradFunctionTests.cs`

## API Design

### Abstract Class: AutogradFunction
```csharp
public abstract class AutogradFunction
{
    protected List<Tensor> SavedTensors { get; }
    protected List<object> SavedScalars { get; }

    protected AutogradFunction();

    // Abstract methods to implement
    public abstract Tensor Forward(params Tensor[] inputs);
    public abstract Tensor[] Backward(Tensor gradOutput);

    // Helper methods
    protected void SaveForBackward(params Tensor[] tensors);
    protected void SaveScalarForBackward(object scalar);
    protected Tensor GetSavedTensor(int index);
    protected T GetSavedScalar<T>(int index);

    // Context management
    public OperationContext CreateContext(string operationName);
    public Tensor Apply(params Tensor[] inputs);
}
```

### Class: FunctionRegistry
```csharp
public static class FunctionRegistry
{
    private static Dictionary<string, Type> _registry;

    public static void Register<T>(string name) where T : AutogradFunction, new();
    public static Type GetFunctionType(string name);
    public static bool IsRegistered(string name);
    public static void Unregister(string name);
}

// Extension method for easy application
public static class TensorFunctionExtensions
{
    public static Tensor ApplyFunction<T>(this Tensor tensor, params object[] args) where T : AutogradFunction, new();
}
```

## Example Usage

### Custom ReLU6 Operation
```csharp
public class ReLU6Function : AutogradFunction
{
    public override Tensor Forward(params Tensor[] inputs)
    {
        Tensor x = inputs[0];
        SaveForBackward(x);
        return Tensor.Clamp(x, 0, 6);
    }

    public override Tensor[] Backward(Tensor gradOutput)
    {
        Tensor x = GetSavedTensor(0);
        Tensor mask = (x > 0) & (x < 6).CastToFloat();
        return new Tensor[] { gradOutput * mask };
    }
}

// Usage
var x = Tensor.Random(10, 10, requiresGrad: true);
var y = new ReLU6Function().Apply(x);
y.Backward();
```

### Custom Loss Function
```csharp
public class FocalLossFunction : AutogradFunction
{
    private double _gamma;
    private Tensor _target;

    public FocalLossFunction(double gamma, Tensor target)
    {
        _gamma = gamma;
        _target = target;
    }

    public override Tensor Forward(params Tensor[] inputs)
    {
        Tensor prediction = inputs[0];
        SaveForBackward(prediction, _target);
        SaveScalarForBackward(_gamma);

        // Focal loss implementation
        Tensor prob = Tensor.Softmax(prediction, dim: 1);
        Tensor focalWeight = Tensor.Pow(1 - prob, _gamma);
        Tensor loss = Tensor.CrossEntropyLoss(prob * focalWeight, _target);
        return loss;
    }

    public override Tensor[] Backward(Tensor gradOutput)
    {
        Tensor prediction = GetSavedTensor(0);
        Tensor target = GetSavedTensor(1);
        double gamma = GetSavedScalar<double>(2);

        // Custom gradient computation
        // ... (implementation)

        return new Tensor[] { grad * gradOutput };
    }
}
```

## Requirements

### Core Functionality
1. **Base Class Infrastructure**
   - Abstract Forward method (takes inputs, returns output)
   - Abstract Backward method (takes gradOutput, returns gradients)
   - Save/Retrieve tensors for backward pass
   - Context management for graph integration

2. **Tensor Saving**
   - Save tensors during forward pass
   - Retrieve tensors during backward pass
   - Automatic memory management
   - Support for scalar values

3. **Graph Integration**
   - Automatically create OperationContext
   - Register with computational graph
   - Handle gradient tracking
   - Support for nested functions

4. **Function Registry**
   - Register custom functions by name
   - Retrieve function type by name
   - Support dynamic function loading

## Implementation Notes

### Memory Management
- Automatically dispose saved tensors after backward pass
- Use weak references where possible
- Clear saved tensors when no longer needed

### Gradient Computation
- Backward receives gradient from upstream
- Should return gradients for each input
- Handle gradient scaling and broadcasting
- Support None gradient for non-differentiable inputs

### Numerical Stability
- Allow users to implement stable gradients
- Handle edge cases manually
- Support gradient clipping

### Error Handling
- Validate tensor shapes
- Check for None gradients
- Provide clear error messages
- Support custom error handling

## Testing Requirements

### Unit Tests
1. Test basic custom function (e.g., square operation)
2. Test tensor saving/retrieval mechanism
3. Test backward gradient computation
4. Test integration with computational graph
5. Test function registry (register/retrieve)
6. Test multiple saved tensors
7. Test scalar saving/retrieval
8. Test error handling (missing inputs, invalid shapes)

### Integration Tests
1. Create custom activation function → verify gradients
2. Create custom loss function → train small model
3. Test nested custom functions
4. Compare gradients with numerical gradients
5. Test custom function with multiple inputs
6. Test custom function with multiple outputs

## Dependencies
- Computational graph infrastructure
- Backward pass implementation
- Tensor operations

## Success Criteria
- Easy-to-use API for custom functions
- Automatic graph integration
- Correct gradient propagation
- Memory-efficient tensor storage
- Clean separation of forward/backward logic
- Extensible for complex operations
