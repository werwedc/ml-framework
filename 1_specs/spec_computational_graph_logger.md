# Spec: Computational Graph Logger

## Overview
Implement logging for computational graphs to visualize model architecture and execution flow, similar to TensorBoard's graph visualization.

## Objectives
- Capture and log model architecture as a directed graph
- Support both static graphs (model definition) and dynamic graphs (execution trace)
- Provide tensor shape annotations and layer connectivity
- Enable gradient flow visualization during backpropagation

## API Design

```csharp
// Graph node types
public enum NodeType
{
    Operation,      // Mathematical operation (e.g., Conv2D, MatMul)
    Tensor,         // Data tensor
    Parameter,      // Model parameter (weight/bias)
    Constant,       // Constant value
    Placeholder     // Input placeholder
}

// Graph node
public class GraphNode
{
    public string Id { get; }
    public string Name { get; }
    public NodeType Type { get; }
    public string OpType { get; } // e.g., "Conv2D", "ReLU"

    // Tensor information
    public long[] Shape { get; }
    public DataType DataType { get; }

    // Graph structure
    public List<string> InputIds { get; }
    public List<string> OutputIds { get; }
    public List<string> ControlDependencies { get; }

    // Metadata
    public Dictionary<string, object> Attributes { get; }
    public Dictionary<string, string> Metadata { get; }
}

// Computational graph
public class ComputationalGraph
{
    public string Name { get; }
    public DateTime Timestamp { get; }
    public long Step { get; }

    public Dictionary<string, GraphNode> Nodes { get; }
    public List<(string from, string to)> Edges { get; }

    // Graph statistics
    public int NodeCount => Nodes.Count;
    public int EdgeCount => Edges.Count;
    public int Depth { get; }
    public int InputCount { get; }
    public int OutputCount { get; }

    public void AddNode(GraphNode node);
    public void AddEdge(string fromId, string toId);
    public IEnumerable<GraphNode> GetInputs();
    public IEnumerable<GraphNode> GetOutputs();
    public IEnumerable<GraphNode> GetTopologicalOrder();
}

// Graph logger interface
public interface IGraphLogger
{
    void LogGraph(ComputationalGraph graph);
    void LogGraph(IModel model);
    Task LogGraphAsync(ComputationalGraph graph);
    Task LogGraphAsync(IModel model);

    // Dynamic graph capture
    void StartGraphCapture(string name);
    void StopGraphCapture();
    ComputationalGraph GetCapturedGraph();

    // Graph analysis
    public GraphAnalysis AnalyzeGraph(string graphId);
}

public class GraphLogger : IGraphLogger
{
    public GraphLogger(IStorageBackend storage);
    public GraphLogger(IEventPublisher eventPublisher);

    // Configuration
    public bool AutoCaptureDynamicGraphs { get; set; } = false;
    public int MaxCaptureDepth { get; set; } = 100;
}

// Graph analysis results
public class GraphAnalysis
{
    public int TotalParameters { get; }
    public int TotalOperations { get; }
    public Dictionary<string, int> OperationCounts { get; }
    public int GraphDepth { get; }
    public int MaxFanIn { get; }
    public int MaxFanOut { get; }

    // Identify potential issues
    public List<string> Warnings { get; }
    public List<string> Recommendations { get; }
}
```

## Implementation Requirements

### 1. GraphNode and GraphEdge (30-45 min)
- Implement `GraphNode` with all properties
- Support various node types (Operation, Tensor, Parameter, etc.)
- Store tensor shape as long[] for flexibility
- Include attributes dictionary for operation-specific parameters
- Add metadata dictionary for extensibility

### 2. ComputationalGraph (45-60 min)
- Implement `ComputationalGraph` class:
  - Store nodes in dictionary by ID
  - Store edges as tuples of node IDs
  - Add node and edge methods with validation
  - Compute graph depth (longest path)
  - Identify input and output nodes
  - Implement topological sort (Kahn's algorithm)
- Implement graph validation:
  - Check for cycles
  - Check for disconnected components
  - Validate node references in edges
- Compute graph statistics (depth, fan-in/fan-out)

### 3. Graph Logger Core (45-60 min)
- Implement `IGraphLogger` interface
- Support logging static graphs:
  - Extract graph from `IModel` interface (placeholder for now)
  - Serialize graph to storage format
- Support logging dynamic graphs:
  - Capture execution trace during forward pass
  - Track actual tensor shapes at runtime
  - Build graph from operation calls
- Implement auto-capture mode:
  - Automatically build graph during execution
  - Limit capture depth to avoid infinite loops
- Integrate with event system (publish `ComputationalGraphEvent`)
- Integrate with storage backend

### 4. Graph Analysis (30-45 min)
- Implement `GraphAnalyzer` class:
  - Count total parameters (sum of all Parameter nodes)
  - Count operations (Operation nodes)
  - Group operations by type
  - Compute graph depth and complexity metrics
  - Identify potential issues:
    - Residual connections
    - Skip connections
    - Very deep layers
    - High fan-in/fan-out (potential bottlenecks)
  - Provide recommendations:
    - Optimization opportunities
    - Architecture improvements
- Run analysis after graph is logged

### 5. Graph Serialization (30-45 min)
- Serialize graph to TensorBoard-compatible format:
  - Use protobuf for graph definition
  - Include node attributes and connections
  - Add tensor shape information
- Support both JSON and binary formats
- Include graph metadata (timestamp, step, etc.)
- Handle circular dependencies (if any)

## File Structure
```
src/
  MLFramework.Visualization/
    Graphs/
      NodeType.cs
      GraphNode.cs
      ComputationalGraph.cs
      IGraphLogger.cs
      GraphLogger.cs
      GraphAnalysis.cs
      Serialization/
        GraphSerializer.cs
        GraphProtobuf.proto (or generated C# classes)

tests/
  MLFramework.Visualization.Tests/
    Graphs/
      GraphLoggerTests.cs
      ComputationalGraphTests.cs
      GraphAnalyzerTests.cs
```

## Dependencies
- `MLFramework.Visualization.Events` (Event system)
- `MLFramework.Visualization.Storage` (Storage backend)

## Integration Points
- Used by TensorBoardVisualizer to log model graphs
- Integrated with model serialization (will be implemented later)
- Data consumed by graph visualization in TensorBoard

## Success Criteria
- Logging graph for ResNet-50 completes in <100ms
- Graph serialization produces valid TensorBoard format
- Graph can be visualized in TensorBoard
- Topological sort correctly orders nodes
- Graph analysis identifies architectural patterns correctly
- Unit tests verify all functionality
