using MLFramework.Visualization.Graphs;

namespace MLFramework.Visualization.Tests.Graphs;

public class GraphAnalysisTests
{
    private readonly GraphAnalyzer _analyzer;

    public GraphAnalysisTests()
    {
        _analyzer = new GraphAnalyzer();
    }

    [Fact]
    public void Analyze_WithSimpleGraph_ReturnsCorrectAnalysis()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var paramNode = new GraphNode("param1", "Weight", NodeType.Parameter);
        var opNode = new GraphNode("op1", "Conv2D", NodeType.Operation, "Conv2D");
        graph.AddNode(paramNode);
        graph.AddNode(opNode);

        // Act
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Equal(1, analysis.TotalParameters);
        Assert.Equal(1, analysis.TotalOperations);
        Assert.True(analysis.OperationCounts.ContainsKey("Conv2D"));
        Assert.Equal(1, analysis.OperationCounts["Conv2D"]);
    }

    [Fact]
    public void Analyze_WithMultipleOperations_ReturnsCorrectCounts()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var op1 = new GraphNode("op1", "Conv2D_1", NodeType.Operation, "Conv2D");
        var op2 = new GraphNode("op2", "Conv2D_2", NodeType.Operation, "Conv2D");
        var op3 = new GraphNode("op3", "ReLU", NodeType.Operation, "ReLU");
        graph.AddNode(op1);
        graph.AddNode(op2);
        graph.AddNode(op3);

        // Act
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Equal(0, analysis.TotalParameters);
        Assert.Equal(3, analysis.TotalOperations);
        Assert.Equal(2, analysis.OperationCounts["Conv2D"]);
        Assert.Equal(1, analysis.OperationCounts["ReLU"]);
    }

    [Fact]
    public void Analyze_WithDeepGraph_AddsDepthWarning()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var previousNode = new GraphNode("input", "Input", NodeType.Placeholder);
        graph.AddNode(previousNode);

        // Create a chain of 101 nodes (depth > 100)
        for (int i = 0; i < 100; i++)
        {
            var node = new GraphNode($"layer_{i}", $"Layer {i}", NodeType.Operation, "Linear");
            graph.AddNode(node);
            graph.AddEdge(previousNode.Id, node.Id);
            previousNode = node;
        }

        // Act
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Contains(analysis.Warnings, w => w.Contains("very deep"));
    }

    [Fact]
    public void Analyze_WithHighFanIn_AddsFanInWarning()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var output = new GraphNode("output", "Output", NodeType.Tensor);

        graph.AddNode(output);

        // Create 11 input nodes (fan-in > 10)
        for (int i = 0; i < 11; i++)
        {
            var input = new GraphNode($"input_{i}", $"Input {i}", NodeType.Placeholder);
            graph.AddNode(input);
            graph.AddEdge(input.Id, output.Id);
        }

        // Act
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Contains(analysis.Warnings, w => w.Contains("high fan-in"));
        Assert.Equal(11, analysis.MaxFanIn);
    }

    [Fact]
    public void Analyze_WithHighFanOut_AddsFanOutWarning()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var input = new GraphNode("input", "Input", NodeType.Placeholder);

        graph.AddNode(input);

        // Create 11 output nodes (fan-out > 10)
        for (int i = 0; i < 11; i++)
        {
            var output = new GraphNode($"output_{i}", $"Output {i}", NodeType.Tensor);
            graph.AddNode(output);
            graph.AddEdge(input.Id, output.Id);
        }

        // Act
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Contains(analysis.Warnings, w => w.Contains("high fan-out"));
        Assert.Equal(11, analysis.MaxFanOut);
    }

    [Fact]
    public void Analyze_WithDisconnectedComponents_AddsWarning()
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
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Contains(analysis.Warnings, w => w.Contains("disconnected components"));
    }

    [Fact]
    public void Analyze_WithCyclicGraph_AddsWarning()
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
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Contains(analysis.Warnings, w => w.Contains("cycle"));
    }

    [Fact]
    public void Analyze_WithSkipConnections_AddsRecommendation()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var input = new GraphNode("input", "Input", NodeType.Placeholder);
        var conv1 = new GraphNode("conv1", "Conv1", NodeType.Operation, "Conv2D");
        var conv2 = new GraphNode("conv2", "Conv2", NodeType.Operation, "Conv2D");
        var add = new GraphNode("add", "Add", NodeType.Operation, "Add");
        var output = new GraphNode("output", "Output", NodeType.Tensor);

        graph.AddNode(input);
        graph.AddNode(conv1);
        graph.AddNode(conv2);
        graph.AddNode(add);
        graph.AddNode(output);

        // Create a skip connection
        graph.AddEdge("input", "conv1");
        graph.AddEdge("conv1", "conv2");
        graph.AddEdge("conv2", "add");
        graph.AddEdge("input", "add"); // Skip connection
        graph.AddEdge("add", "output");

        // Act
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Contains(analysis.Recommendations, r => r.Contains("skip connection"));
    }

    [Fact]
    public void Analyze_WithResidualConnections_AddsRecommendation()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var input = new GraphNode("input", "Input", NodeType.Placeholder);
        var conv1 = new GraphNode("conv1", "Conv1", NodeType.Operation, "Conv2D");
        var conv2 = new GraphNode("conv2", "Conv2", NodeType.Operation, "Conv2D");
        var add = new GraphNode("add", "Add", NodeType.Operation, "Add");
        var output = new GraphNode("output", "Output", NodeType.Tensor);

        graph.AddNode(input);
        graph.AddNode(conv1);
        graph.AddNode(conv2);
        graph.AddNode(add);
        graph.AddNode(output);

        // Create a residual connection
        graph.AddEdge("input", "conv1");
        graph.AddEdge("conv1", "conv2");
        graph.AddEdge("conv2", "add");
        graph.AddEdge("input", "add"); // Residual connection
        graph.AddEdge("add", "output");

        // Act
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Contains(analysis.Recommendations, r => r.Contains("residual"));
    }

    [Fact]
    public void Analyze_WithConv2DAndBatchNorm_AddsRecommendation()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var conv = new GraphNode("conv", "Conv2D", NodeType.Operation, "Conv2D");
        var bn = new GraphNode("bn", "BatchNorm", NodeType.Operation, "BatchNorm");
        graph.AddNode(conv);
        graph.AddNode(bn);
        graph.AddEdge("conv", "bn");

        // Act
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Contains(analysis.Recommendations, r => r.Contains("Conv2D + BatchNorm"));
    }

    [Fact]
    public void Analyze_WithDropout_AddsRecommendation()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var dropout = new GraphNode("dropout", "Dropout", NodeType.Operation, "Dropout");
        graph.AddNode(dropout);

        // Act
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Contains(analysis.Recommendations, r => r.Contains("Dropout"));
    }

    [Fact]
    public void Analyze_WithPoolingLayers_AddsRecommendation()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var pool = new GraphNode("pool", "MaxPool2D", NodeType.Operation, "MaxPool2D");
        graph.AddNode(pool);

        // Act
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Contains(analysis.Recommendations, r => r.Contains("pooling"));
    }

    [Fact]
    public void Analyze_WithActivationFunctions_AddsRecommendation()
    {
        // Arrange
        var graph = new ComputationalGraph("test_graph", 0);
        var relu = new GraphNode("relu", "ReLU", NodeType.Operation, "ReLU");
        graph.AddNode(relu);

        // Act
        var analysis = _analyzer.Analyze(graph);

        // Assert
        Assert.Contains(analysis.Recommendations, r => r.Contains("activation"));
    }

    [Fact]
    public void Analyze_WithNullGraph_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _analyzer.Analyze(null!));
    }

    [Fact]
    public void GraphAnalysis_Properties_AreCorrectlySet()
    {
        // Arrange
        var operationCounts = new Dictionary<string, int>
        {
            { "Conv2D", 2 },
            { "ReLU", 1 }
        };

        // Act
        var analysis = new GraphAnalysis(
            1000,  // Total parameters
            3,     // Total operations
            operationCounts,
            5,     // Graph depth
            2,     // Max fan-in
            3,     // Max fan-out
            new List<string> { "Warning 1" },
            new List<string> { "Recommendation 1" });

        // Assert
        Assert.Equal(1000, analysis.TotalParameters);
        Assert.Equal(3, analysis.TotalOperations);
        Assert.Equal(2, analysis.OperationCounts["Conv2D"]);
        Assert.Equal(1, analysis.OperationCounts["ReLU"]);
        Assert.Equal(5, analysis.GraphDepth);
        Assert.Equal(2, analysis.MaxFanIn);
        Assert.Equal(3, analysis.MaxFanOut);
        Assert.Single(analysis.Warnings);
        Assert.Equal("Warning 1", analysis.Warnings[0]);
        Assert.Single(analysis.Recommendations);
        Assert.Equal("Recommendation 1", analysis.Recommendations[0]);
    }
}
