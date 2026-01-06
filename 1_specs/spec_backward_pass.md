# Spec: Basic Backward Pass Implementation

## Overview
Implement the reverse-mode automatic differentiation engine that traverses the computational graph and computes gradients via the chain rule.

## Files to Create
- `src/MLFramework/Autograd/BackwardPass.cs`
- `src/MLFramework/Autograd/GradientComputer.cs`
- `tests/MLFramework.Tests/Autograd/BackwardPassTests.cs`

## API Design

### Class: BackwardPass
```csharp
public class BackwardPass
{
    public GraphBuilder Graph { get; }
    public bool RetainGraph { get; set; }

    public BackwardPass(GraphBuilder graph);
    public void Run(Tensor lossTensor, Tensor? gradient = null);
    public void RunFromNode(GraphNode node, Tensor? gradient = null);
    private void ComputeGradients(GraphNode node, Tensor gradOutput);
    private void PropagateToChildren(GraphNode node, Tensor[] gradients);
}
```

### Class: GradientComputer
```csharp
public static class GradientComputer
{
    public static Tensor ComputeGradient(Tensor gradOutput, Tensor input, OperationContext context);
    public static Tensor[] ComputeGradients(Tensor gradOutput, Tensor[] inputs, OperationContext context);
    public static Tensor NumericalGradient(Func<Tensor, Tensor> f, Tensor x, double epsilon = 1e-6);
}
```

## Requirements

### Core Functionality
1. **Reverse-Mode Differentiation**
   - Start from loss tensor with gradient = 1.0
   - Traverse graph in reverse (post-order DFS)
   - Apply chain rule at each node
   - Accumulate gradients at leaf nodes

2. **Gradient Propagation**
   - Call backward function from OperationContext
   - Retrieve saved intermediate tensors
   - Pass gradient to child nodes
   - Handle multiple gradient paths (sum at convergence)

3. **Graph Retention**
   - Option to retain graph after backward pass
   - By default, clear graph and saved tensors
   - Support for higher-order derivatives (retain graph)

4. **Gradient Initialization**
   - Loss tensor gets gradient = 1.0 by default
   - Support custom initial gradient
   - Handle scalar and tensor gradients

## Implementation Notes

### Traversal Algorithm
- Use stack-based DFS for graph traversal
- Topological sort ensures correct order
- Handle multiple children in parallel where possible
- Early termination for detached tensors

### Chain Rule Implementation
- At each node: dL/dx = dL/dy * dy/dx
- Backward function returns dy/dx for each input
- Gradient accumulation: grad += newGrad
- Handle broadcasting and shape mismatches

### Memory Management
- Clear saved tensors after use (unless retain graph)
- Dispose intermediate gradient tensors
- Reference counting for graph nodes
- Lazy gradient tensor allocation

### Error Handling
- Detect cycles in graph
- Validate gradient shapes
- Handle non-differentiable operations gracefully
- Provide clear error messages

## Testing Requirements

### Unit Tests
1. Compute gradients for simple scalar chain (y = x + 1, L = y²)
2. Test gradient propagation through multiple operations
3. Test gradient accumulation at convergence points
4. Test graph retention flag → verify graph preserved
5. Test graph non-retention → verify graph cleared
6. Test custom initial gradient → verify correct computation
7. Test backward from intermediate node (not loss)

### Integration Tests
1. Compute gradients for linear regression model
2. Verify gradients against numerical gradients
3. Test gradient computation for deep network (10+ layers)
4. Test gradient for operations with multiple inputs
5. Test gradient for operations with multiple outputs

## Dependencies
- Computational graph infrastructure
- Operation registry system
- Tensor gradient tracking
- Memory management system

## Success Criteria
- Accurate gradient computation (within 1e-6 of numerical gradients)
- Handles computational graphs of arbitrary depth
- Efficient memory usage (clears intermediate tensors)
- Supports higher-order derivatives (retain graph)
- Clear error messages for invalid operations
