# Spec: Computational Graph Builder

## Overview
Implement the computational graph infrastructure that automatically builds a dynamic graph during forward propagation, tracking tensor dependencies and operation history.

## Files to Create
- `src/MLFramework/Autograd/GraphNode.cs`
- `src/MLFramework/Autograd/OperationContext.cs`
- `src/MLFramework/Autograd/GraphBuilder.cs`
- `tests/MLFramework.Tests/Autograd/GraphBuilderTests.cs`

## API Design

### Class: GraphNode
```csharp
public class GraphNode
{
    public Tensor OutputTensor { get; }
    public IReadOnlyList<GraphNode> Children { get; }
    public OperationContext Operation { get; }
    public bool IsLeaf { get; }
    public int GradFnId { get; }

    public GraphNode(Tensor output, OperationContext operation, params GraphNode[] children);
    public void Register();
    public void Dispose();
}
```

### Class: OperationContext
```csharp
public class OperationContext
{
    public string OperationName { get; }
    public Dictionary<string, object> SavedTensors { get; }
    public Func<Tensor, Tensor[]> BackwardFn { get; }
    public int OperationId { get; }

    public OperationContext(string name, Func<Tensor, Tensor[]> backwardFn);
    public void SaveTensor(string key, Tensor tensor);
    public Tensor GetSavedTensor(string key);
    public void ClearSavedTensors();
}
```

### Class: GraphBuilder
```csharp
public class GraphBuilder
{
    public bool IsEnabled { get; set; }
    public GraphNode CurrentNode { get; private set; }
    public Stack<GraphNode> NodeStack { get; }

    public GraphBuilder();
    public GraphNode CreateNode(Tensor output, OperationContext operation, params GraphNode[] inputs);
    public void PushScope(GraphNode node);
    public GraphNode PopScope();
    public void ClearGraph();
    public List<GraphNode> GetRootNodes();
}
```

## Requirements

### Core Functionality
1. **Graph Construction**
   - Automatically create node for each operation requiring gradients
   - Track parent-child relationships between tensors
   - Maintain DAG structure (no cycles)

2. **Operation Context**
   - Store operation metadata (name, ID)
   - Save intermediate tensors needed for backward pass
   - Store backward function reference
   - Clear saved tensors after backward pass

3. **Dynamic Graph Management**
   - Build graph incrementally during forward pass
   - Support graph clearing between training iterations
   - Enable/disable graph building (for inference mode)

4. **Scope Management**
   - Push/pop context for nested operations
   - Track current operation context
   - Handle complex operation chains

## Implementation Notes

### Memory Efficiency
- Only build graph when `requiresGrad` is true for at least one input
- Use weak references for saved tensors when possible
- Clear saved tensors after backward pass
- Dispose graph nodes when no longer needed

### Performance
- Minimize overhead during forward pass
- Use efficient data structures for node storage
- Lazy node creation for leaf tensors
- Consider object pooling for OperationContext

### Thread Safety
- Thread-local graph builder instances
- Safe for multi-threaded forward passes
- Each thread maintains independent graph

## Testing Requirements

### Unit Tests
1. Create simple operation chain (x → y → z) → verify graph structure
2. Test graph building with `requiresGrad: false` → verify no nodes created
3. Test graph building with `requiresGrad: true` → verify nodes created
4. Test graph clearing → verify all nodes disposed
5. Test saved tensor storage → verify tensors retrievable
6. Test scope push/pop → verify context management
7. Test multi-threaded graph building → verify independent graphs

### Integration Tests
1. Build graph for neural network forward pass → verify complete DAG
2. Verify no cycles in graph
3. Test graph size for large models → verify memory efficiency

## Dependencies
- Tensor gradient tracking infrastructure
- Core Tensor operations
- Memory management system

## Success Criteria
- Graph automatically builds during forward pass
- Minimal overhead (< 5%) for non-gradient tensors
- Memory-efficient intermediate storage
- Thread-safe for multi-threaded forward passes
- Clean disposal of graph resources
