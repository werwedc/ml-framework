# Spec: CustomFunction Base Class

## Overview
Implement the abstract `CustomFunction` base class that defines the contract for custom autograd functions. This class provides the API for users to define custom forward and backward passes.

## Requirements

### 1. Class Definition
Create an abstract `CustomFunction` class in `src/Autograd/CustomFunction.cs` with:
- Abstract `Forward` method for computing outputs
- Abstract `Backward` method for computing gradients
- Instance `Apply` method to invoke the function
- Connection to the autograd computational graph

### 2. Core Methods

#### abstract Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
- Abstract method that subclasses must implement
- Receives input tensors and a FunctionContext
- Computes the forward pass
- Should save necessary tensors/objects to ctx for backward pass
- Returns array of output tensors (can be empty)
- Called internally by Apply()

#### abstract Tensor[] Backward(Tensor[] grad_outputs, FunctionContext ctx)
- Abstract method that subclasses must implement
- Receives gradients of outputs and the saved context
- Retrieves saved state from ctx (saved during forward)
- Computes gradients with respect to inputs
- Returns array of gradient tensors (one per input, can be null)
- Called internally during backward propagation

#### Tensor Apply(params Tensor[] inputs)
- Public instance method to invoke the custom function
- Creates new FunctionContext for this invocation
- Calls Forward() with the inputs and context
- Registers the function invocation with the autograd engine
- Returns the first output tensor (for single-output convenience)
- For multiple outputs, provide an overloaded version: `Tensor[] ApplyMany(params Tensor[] inputs)`

### 3. Autograd Integration

The function must integrate with the autograd engine:
- Create a node in the computational graph
- Store the context with the node for later backward pass
- Propagate requires_grad property from inputs to outputs
- Handle gradient flow during backward pass

Integration points:
- When Apply() is called, create a `CustomFunctionNode` in the graph
- Store reference to the CustomFunction instance and context
- During backward, call Backward() with the incoming gradients

### 4. Output Handling

#### Single Output (Tensor Apply(params Tensor[] inputs))
- Returns the first output tensor (inputs[0])
- Throws InvalidOperationException if no outputs are returned
- Convenience method for common single-output functions

#### Multiple Outputs (Tensor[] ApplyMany(params Tensor[] inputs))
- Returns all output tensors
- Returns empty array if no outputs (for side-effect functions)

### 5. Input Validation
- Validate inputs array is not null
- Validate each input tensor is not null
- Validate context passed to Forward/Backward is not null

### 6. Error Handling
- If Forward returns null, throw with message: "Forward pass returned null"
- If Backward returns null, throw with message: "Backward pass returned null"
- If gradient shapes don't match, throw with message: "Gradient at index {i} has shape {actualShape} but expected {expectedShape}"

## Implementation Notes

### Apply Method Implementation
```csharp
public Tensor Apply(params Tensor[] inputs)
{
    var outputs = ApplyMany(inputs);
    if (outputs.Length == 0)
        throw new InvalidOperationException("Function produced no outputs");

    return outputs[0];
}

public Tensor[] ApplyMany(params Tensor[] inputs)
{
    // Validate inputs
    if (inputs == null || inputs.Any(t => t == null))
        throw new ArgumentNullException(nameof(inputs));

    // Create context
    var ctx = new FunctionContext();

    // Call forward
    var outputs = Forward(inputs, ctx);
    if (outputs == null)
        throw new InvalidOperationException("Forward pass returned null");

    // Register with autograd engine
    RegisterWithAutograd(inputs, outputs, ctx);

    return outputs;
}
```

### Autograd Registration
The `RegisterWithAutograd` method should:
1. Create a backward function that calls this.Backward() with saved context
2. Attach backward function to each output tensor that requires grad
3. Ensure context is accessible during backward pass

## Testing Requirements
Create unit tests in `tests/Autograd/CustomFunctionTests.cs`:

1. **Basic Functionality Tests**
   - Create a simple subclass (e.g., IdentityFunction) that returns inputs unchanged
   - Test Apply() returns the first output
   - Test ApplyMany() returns all outputs

2. **Context Tests**
   - Verify a fresh context is created for each Apply() call
   - Verify context is passed correctly to Forward()
   - Verify context is accessible during backward pass

3. **Error Handling Tests**
   - Test with null inputs array
   - Test with null individual tensors
   - Test Forward returning null
   - Test Backward returning null
   - Test gradient shape mismatches

4. **Autograd Integration Tests**
   - Create a simple function and verify it connects to the graph
   - Verify requires_grad propagates from inputs to outputs
   - Verify backward pass calls Backward() correctly

## Success Criteria
- [ ] CustomFunction abstract class is implemented in `src/Autograd/CustomFunction.cs`
- [ ] Forward and Backward methods are abstract
- [ ] Apply() and ApplyMany() methods work correctly
- [ ] Autograd integration is functional
- [ ] All error handling is in place
- [ ] Unit tests cover all scenarios with >90% code coverage
