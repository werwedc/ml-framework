using MLFramework.Visualization.Graphs;

namespace MLFramework.Visualization.Tests.Graphs;

public class ComputationalGraphTests
{
    [Fact]
    public void Constructor_WithValidParameters_CreatesGraph()
    {
        // Arrange & Act
        var graph = new ComputationalGraph("test_graph", 0);

        // Assert
        Assert.Equal("test_graph", graph.Name);
        Assert.Equal(0, graph.Step);
        Assert.Empty(graph.Nodes);
        Assert.Empty(graph.Edges);
        Assert.Equal(0, graph.NodeCount);
        Assert.Equal(0, graph.EdgeCount);
        Assert.Equal(0, graph.Depth);
        Assert.Equal(0, graph.InputCount);
        Assert.Equal(0, graph.OutputCount);
    }

    [Fact]
    public void Constructor_WithNullName_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ComputationalGraph(null!, 0));
    }

    [Fact]
    public void AddNode_WithValidNode_AddsNodeToGraph()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");

        // Act
        graph.AddNode(node);

        // Assert
        Assert.Single(graph.Nodes);
        Assert.True(graph.Nodes.ContainsKey("node1"));
        Assert.Equal(node, graph.Nodes["node1"]);
    }

    [Fact]
    public void AddNode_WithDuplicateId_ThrowsArgumentException()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node1 = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        var node2 = new GraphNode("node1", "Node 2", NodeType.Operation, "Conv2D");

        // Act
        graph.AddNode(node1);

        // Assert
        Assert.Throws<ArgumentException>(() => graph.AddNode(node2));
    }

    [Fact]
    public void AddNode_WithNullNode_ThrowsArgumentNullException()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => graph.AddNode(null!));
    }

    [Fact]
    public void AddEdge_WithValidNodes_AddsEdgeToGraph()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node1 = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        var node2 = new GraphNode("node2", "Node 2", NodeType.Operation, "ReLU");
        graph.AddNode(node1);
        graph.AddNode(node2);

        // Act
        graph.AddEdge("node1", "node2");

        // Assert
        Assert.Single(graph.Edges);
        Assert.Equal(("node1", "node2"), graph.Edges[0]);
        Assert.Contains("node2", node1.OutputIds);
        Assert.Contains("node1", node2.InputIds);
    }

    [Fact]
    public void AddEdge_WithNonExistentFromNode_ThrowsArgumentException()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node2 = new GraphNode("node2", "Node 2", NodeType.Operation, "ReLU");
        graph.AddNode(node2);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => graph.AddEdge("node1", "node2"));
    }

    [Fact]
    public void AddEdge_WithNonExistentToNode_ThrowsArgumentException()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node1 = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        graph.AddNode(node1);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => graph.AddEdge("node1", "node2"));
    }

    [Fact]
    public void AddEdge_WithDuplicateEdge_ThrowsArgumentException()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node1 = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        var node2 = new GraphNode("node2", "Node 2", NodeType.Operation, "ReLU");
        graph.AddNode(node1);
        graph.AddNode(node2);
        graph.AddEdge("node1", "node2");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => graph.AddEdge("node1", "node2"));
    }

    [Fact]
    public void GetInputs_WithInputNodes_ReturnsCorrectNodes()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var inputNode = new GraphNode("input", "Input", NodeType.Placeholder);
        var opNode = new GraphNode("op", "Operation", NodeType.Operation, "Conv2D");
        graph.AddNode(inputNode);
        graph.AddNode(opNode);
        graph.AddEdge("input", "op");

        // Act
        var inputs = graph.GetInputs();

        // Assert
        Assert.Single(inputs);
        Assert.Equal(inputNode, inputs.First());
    }

    [Fact]
    public void GetOutputs_WithOutputNodes_ReturnsCorrectNodes()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var opNode = new GraphNode("op", "Operation", NodeType.Operation, "Conv2D");
        var outputNode = new GraphNode("output", "Output", NodeType.Tensor);
        graph.AddNode(opNode);
        graph.AddNode(outputNode);
        graph.AddEdge("op", "output");

        // Act
        var outputs = graph.GetOutputs();

        // Assert
        Assert.Single(outputs);
        Assert.Equal(outputNode, outputs.First());
    }

    [Fact]
    public void GetTopologicalOrder_WithLinearGraph_ReturnsCorrectOrder()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node1 = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        var node2 = new GraphNode("node2", "Node 2", NodeType.Operation, "ReLU");
        var node3 = new GraphNode("node3", "Node 3", NodeType.Operation, "Linear");
        graph.AddNode(node1);
        graph.AddNode(node2);
        graph.AddNode(node3);
        graph.AddEdge("node1", "node2");
        graph.AddEdge("node2", "node3");

        // Act
        var topologicalOrder = graph.GetTopologicalOrder().ToList();

        // Assert
        Assert.Equal(3, topologicalOrder.Count);
        Assert.Equal(node1, topologicalOrder[0]);
        Assert.Equal(node2, topologicalOrder[1]);
        Assert.Equal(node3, topologicalOrder[2]);
    }

    [Fact]
    public void GetTopologicalOrder_WithCycle_ThrowsInvalidOperationException()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node1 = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        var node2 = new GraphNode("node2", "Node 2", NodeType.Operation, "ReLU");
        graph.AddNode(node1);
        graph.AddNode(node2);
        graph.AddEdge("node1", "node2");
        graph.AddEdge("node2", "node1");

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => graph.GetTopologicalOrder());
    }

    [Fact]
    public void HasCycle_WithAcyclicGraph_ReturnsFalse()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node1 = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        var node2 = new GraphNode("node2", "Node 2", NodeType.Operation, "ReLU");
        graph.AddNode(node1);
        graph.AddNode(node2);
        graph.AddEdge("node1", "node2");

        // Act
        var hasCycle = graph.HasCycle();

        // Assert
        Assert.False(hasCycle);
    }

    [Fact]
    public void HasCycle_WithCyclicGraph_ReturnsTrue()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node1 = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        var node2 = new GraphNode("node2", "Node 2", NodeType.Operation, "ReLU");
        graph.AddNode(node1);
        graph.AddNode(node2);
        graph.AddEdge("node1", "node2");
        graph.AddEdge("node2", "node1");

        // Act
        var hasCycle = graph.HasCycle();

        // Assert
        Assert.True(hasCycle);
    }

    [Fact]
    public void GetDisconnectedComponents_WithConnectedGraph_ReturnsSingleComponent()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node1 = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        var node2 = new GraphNode("node2", "Node 2", NodeType.Operation, "ReLU");
        graph.AddNode(node1);
        graph.AddNode(node2);
        graph.AddEdge("node1", "node2");

        // Act
        var components = graph.GetDisconnectedComponents();

        // Assert
        Assert.Single(components);
        Assert.Equal(2, components[0].Count);
    }

    [Fact]
    public void GetDisconnectedComponents_WithDisconnectedGraph_ReturnsMultipleComponents()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node1 = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        var node2 = new GraphNode("node2", "Node 2", NodeType.Operation, "ReLU");
        var node3 = new GraphNode("node3", "Node 3", NodeType.Operation, "Linear");
        graph.AddNode(node1);
        graph.AddNode(node2);
        graph.AddNode(node3);
        graph.AddEdge("node1", "node2");

        // Act
        var components = graph.GetDisconnectedComponents();

        // Assert
        Assert.Equal(2, components.Count);
        Assert.Equal(2, components[0].Count);
        Assert.Single(components[1]);
    }

    [Fact]
    public void Depth_WithLinearGraph_CalculatesCorrectDepth()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var node1 = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        var node2 = new GraphNode("node2", "Node 2", NodeType.Operation, "ReLU");
        var node3 = new GraphNode("node3", "Node 3", NodeType.Operation, "Linear");
        graph.AddNode(node1);
        graph.AddNode(node2);
        graph.AddNode(node3);
        graph.AddEdge("node1", "node2");
        graph.AddEdge("node2", "node3");

        // Act
        var depth = graph.Depth;

        // Assert
        Assert.Equal(2, depth); // Depth is number of edges on the longest path
    }

    [Fact]
    public void Depth_WithBranchedGraph_CalculatesCorrectDepth()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var input = new GraphNode("input", "Input", NodeType.Placeholder);
        var op1 = new GraphNode("op1", "Op 1", NodeType.Operation, "Conv2D");
        var op2 = new GraphNode("op2", "Op 2", NodeType.Operation, "Conv2D");
        var op3 = new GraphNode("op3", "Op 3", NodeType.Operation, "ReLU");
        var output = new GraphNode("output", "Output", NodeType.Tensor);
        graph.AddNode(input);
        graph.AddNode(op1);
        graph.AddNode(op2);
        graph.AddNode(op3);
        graph.AddNode(output);
        graph.AddEdge("input", "op1");
        graph.AddEdge("input", "op2");
        graph.AddEdge("op1", "op3");
        graph.AddEdge("op2", "op3");
        graph.AddEdge("op3", "output");

        // Act
        var depth = graph.Depth;

        // Assert
        Assert.Equal(2, depth); // Longest path is input -> op1 -> op3 -> output (3 edges)
    }
}
