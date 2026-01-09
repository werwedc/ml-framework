# Spec: Autograd Integration for Custom Functions

## Overview
Implement the integration layer that connects custom functions to the existing autograd computational graph. This spec covers the mechanism to register custom functions, create graph nodes, and ensure gradients flow correctly through custom backward passes.

## Requirements

### 1. Graph Node for Custom Functions

Create a `CustomFunctionNode` class in `src/Autograd/Nodes/CustomFunctionNode.cs` that represents a custom function in the computational graph.

#### Properties
- `CustomFunction Function` - Reference to the custom function instance
- `FunctionContext Context` - Saved context from forward pass
- `Tensor[] Inputs` - Input tensors to the function
- `Tensor[] Outputs` - Output tensors from the function
- `Guid Id` - Unique identifier for debugging

#### Methods
- `void Backward(Tensor[] gradOutputs)` - Execute backward pass
- `void Dispose()` - Cleanup resources and context

### 2. Autograd Engine Extensions

Extend the existing autograd engine to support custom functions. Create or modify classes in `src/Autograd/`:

#### RegisterCustomFunction Method
```csharp
public void RegisterCustomFunction(
    Tensor[] outputs,
    CustomFunction function,
    FunctionContext context,
    Tensor[] inputs)
{
    var node = new CustomFunctionNode
    {
        Function = function,
        Context = context,
        Inputs = inputs,
        Outputs = outputs
    };

    // Attach backward function to outputs that require grad
    for (int i = 0; i < outputs.Length; i++)
    {
        if (outputs[i].RequiresGrad)
        {
            outputs[i].GradFn = CreateBackwardFunction(node, i);
        }
    }

    // Add node to graph
    AddNode(node);
}
```

#### CreateBackwardFunction Method
```csharp
private Func<Tensor[], Tensor[]> CreateBackwardFunction(CustomFunctionNode node, int outputIndex)
{
    return (Tensor[] upstreamGrads) =>
    {
        // Call the custom function's backward pass
        var gradOutputs = new Tensor[node.Outputs.Length];
        gradOutputs[outputIndex] = upstreamGrads[0];

        var gradInputs = node.Function.Backward(gradOutputs, node.Context);

        return gradInputs;
    };
}
```

### 3. Tensor Extensions

Add necessary properties/methods to the Tensor class (in `src/Tensors/Tensor.cs` or `src/Autograd/TensorExtensions.cs`):

#### Properties
- `GradFn` - Reference to backward function (or null)
- `Grad` - Computed gradient tensor
- `RequiresGrad` - Whether gradient is needed

#### Methods
- `void Backward()` - Compute gradients backward through the graph
- `void AccumulateGrad(Tensor grad)` - Add gradient to tensor's grad

### 4. Context Lifecycle Management

Ensure proper lifecycle management for FunctionContext:

#### During Forward Pass
1. Create new FunctionContext for each invocation
2. Pass context to Forward() method
3. Forward() saves necessary tensors/objects
4. Store context with graph node

#### During Backward Pass
1. Retrieve context from graph node
2. Pass context to Backward() method
3. Backward() retrieves saved tensors/objects
4. Dispose context after backward pass (or clear)

#### Automatic Cleanup
- Implement IDisposable on CustomFunctionNode
- Clear context when node is disposed
- Optionally clear context immediately after backward pass

### 5. Gradient Accumulation

Support gradient accumulation when a tensor is used multiple times:

#### Accumulation Logic
```csharp
public void AccumulateGrad(Tensor newGrad)
{
    if (Grad == null)
    {
        Grad = newGrad.Clone();
    }
    else
    {
        Grad += newGrad;
    }
}
```

#### Multiple Use Cases
If a tensor is used as input to multiple functions, gradients are accumulated:
- Sum all incoming gradients
- Each function contributes one gradient

### 6. Backward Traversal

Implement backward traversal logic in the autograd engine:

#### Algorithm
1. Start from output tensors that called Backward()
2. Follow graph edges backwards
3. For each node, call its backward function
4. Pass gradients to previous nodes
5. Accumulate gradients at leaf nodes

#### Pseudo-code
```csharp
public void Backward(Tensor output, Tensor initialGrad = null)
{
    var queue = new Queue<Tensor>();
    var visited = new HashSet<Guid>();

    if (initialGrad != null)
    {
        output.AccumulateGrad(initialGrad);
    }
    else
    {
        output.AccumulateGrad(Tensor.OnesLike(output));
    }

    queue.Enqueue(output);
    visited.Add(output.Id);

    while (queue.Count > 0)
    {
        var tensor = queue.Dequeue();
        var gradFn = tensor.GradFn;

        if (gradFn != null)
        {
            // Compute gradients w.r.t. inputs
            var inputGrads = gradFn(new[] { tensor.Grad });

            // Propagate to input tensors
            for (int i = 0; i < inputGrads.Length; i++)
            {
                var inputGrad = inputGrads[i];
                var inputTensor = tensor.GradFn.Inputs[i];

                if (inputTensor.RequiresGrad)
                {
                    inputTensor.AccumulateGrad(inputGrad);

                    if (!visited.Contains(inputTensor.Id))
                    {
                        visited.Add(inputTensor.Id);
                        queue.Enqueue(inputTensor);
                    }
                }
            }
        }
    }
}
```

### 7. Error Handling

#### Graph Validation
- Detect cycles in the computational graph
- Ensure all tensors are properly connected
- Validate gradient flow (no disconnected nodes)

#### Context Validation
- Ensure context is not null when accessing in backward pass
- Ensure context is not already disposed
- Ensure saved tensors are not disposed

#### Gradient Validation
- Validate gradient shapes match input shapes
- Validate gradient dtypes match input dtypes
- Use GradientValidator from previous spec

### 8. Debugging Support

#### Graph Visualization (Optional)
- Provide method to print/visualize the computational graph
- Show custom function nodes with their type and ID
- Show tensor connections

#### Tracing (Optional)
- Provide option to trace forward/backward passes
- Log function calls and gradient values
- Useful for debugging custom functions

## Implementation Notes

### Integration with CustomFunction.Apply
Update the `Apply` method in CustomFunction to use the autograd integration:

```csharp
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
    AutogradEngine.Instance.RegisterCustomFunction(outputs, this, ctx, inputs);

    return outputs;
}
```

### Thread Safety
If supporting multi-threaded computation:
- Use thread-local storage for context
- Ensure gradient accumulation is atomic
- Use proper locking for graph modifications

## Testing Requirements
Create unit tests in `tests/Autograd/CustomFunctionIntegrationTests.cs`:

1. **Basic Integration Tests**
   - Create a simple custom function
   - Apply to tensors and verify graph node is created
   - Call Backward() and verify gradients flow
   - Verify context is passed correctly

2. **Multiple Output Tests**
   - Test function with multiple outputs
   - Verify gradients flow to correct inputs
   - Verify gradient accumulation

3. **Graph Construction Tests**
   - Build a computational graph with multiple custom functions
   - Verify graph structure is correct
   - Verify backward traversal order

4. **Context Lifecycle Tests**
   - Verify context is created for each forward pass
   - Verify context is accessible during backward pass
   - Verify context is disposed after backward pass
   - Verify context cleanup prevents memory leaks

5. **Gradient Accumulation Tests**
   - Use same tensor as input to multiple functions
   - Verify gradients are accumulated correctly
   - Verify final gradient is sum of all contributions

6. **Error Handling Tests**
   - Test with invalid graph (cycles, disconnected nodes)
   - Test with null context
   - Test with gradient shape mismatches

7. **Integration with Existing Autograd Tests**
   - Ensure custom functions work alongside standard operations
   - Test mixed graphs with custom and built-in functions

## Success Criteria
- [ ] CustomFunctionNode class is implemented
- [ ] AutogradEngine supports custom function registration
- [ ] CustomFunction.Apply integrates with autograd engine
- [ ] Context lifecycle is managed correctly
- [ ] Backward traversal works with custom functions
- [ ] Gradient accumulation works for multiple uses
- [ ] All error handling is in place
- [ ] Unit tests cover all scenarios with >90% code coverage
- [ ] All example functions work in computational graphs
