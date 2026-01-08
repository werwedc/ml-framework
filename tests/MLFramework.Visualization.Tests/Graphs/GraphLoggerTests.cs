using MachineLearning.Visualization.Events;
using MachineLearning.Visualization.Storage;
using MLFramework.Visualization.Graphs;

namespace MLFramework.Visualization.Tests.Graphs;

public class GraphLoggerTests
{
    private class MockStorageBackend : IStorageBackend
    {
        private readonly List<Event> _storedEvents = new();

        public List<Event> StoredEvents => _storedEvents;

        public Task StoreAsync(Event eventData, CancellationToken cancellationToken = default)
        {
            _storedEvents.Add(eventData);
            return Task.CompletedTask;
        }

        public Task StoreBatchAsync(System.Collections.Immutable.ImmutableArray<Event> events, CancellationToken cancellationToken = default)
        {
            foreach (var evt in events)
            {
                _storedEvents.Add(evt);
            }
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            _storedEvents.Clear();
        }
    }

    private class MockModel : IModel
    {
        public string Name { get; set; } = "MockModel";
        private readonly ComputationalGraph _graph;

        public MockModel(ComputationalGraph graph)
        {
            _graph = graph;
        }

        public ComputationalGraph GetGraph()
        {
            return _graph;
        }
    }

    [Fact]
    public void Constructor_WithValidStorage_CreatesLogger()
    {
        // Arrange
        var storage = new MockStorageBackend();

        // Act
        var logger = new GraphLogger(storage);

        // Assert
        Assert.NotNull(logger);
        Assert.Equal(0, logger.GetAllGraphs().Count);
    }

    [Fact]
    public void Constructor_WithNullStorage_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new GraphLogger(null!));
    }

    [Fact]
    public void LogGraph_WithValidGraph_LogsGraph()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);
        var graph = new ComputationalGraph("test_graph", 0);
        var node = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        graph.AddNode(node);

        // Act
        logger.LogGraph(graph);

        // Assert
        Assert.Single(logger.GetAllGraphs());
        Assert.Single(storage.StoredEvents);
        Assert.IsType<ComputationalGraphEvent>(storage.StoredEvents[0]);
    }

    [Fact]
    public void LogGraph_WithNullGraph_ThrowsArgumentNullException()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => logger.LogGraph(null!));
    }

    [Fact]
    public void LogGraph_WithModel_LogsModelGraph()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);
        var graph = new ComputationalGraph("test_graph", 0);
        var node = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        graph.AddNode(node);
        var model = new MockModel(graph);

        // Act
        logger.LogGraph(model);

        // Assert
        Assert.Single(logger.GetAllGraphs());
        Assert.Single(storage.StoredEvents);
    }

    [Fact]
    public void LogGraph_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => logger.LogGraph(null!));
    }

    [Fact]
    public async Task LogGraphAsync_WithValidGraph_LogsGraph()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);
        var graph = new ComputationalGraph("test_graph", 0);
        var node = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        graph.AddNode(node);

        // Act
        await logger.LogGraphAsync(graph);

        // Assert
        Assert.Single(logger.GetAllGraphs());
        Assert.Single(storage.StoredEvents);
    }

    [Fact]
    public async Task LogGraphAsync_WithNullGraph_ThrowsArgumentNullException()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() => logger.LogGraphAsync(null!));
    }

    [Fact]
    public void StartGraphCapture_WithValidName_StartsCapture()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);

        // Act
        logger.StartGraphCapture("captured_graph");

        // Assert
        Assert.NotNull(logger.GetCapturedGraph());
        Assert.Equal("captured_graph", logger.GetCapturedGraph()!.Name);
    }

    [Fact]
    public void StartGraphCapture_WithEmptyName_ThrowsArgumentException()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => logger.StartGraphCapture(""));
    }

    [Fact]
    public void StartGraphCapture_WithActiveCapture_ThrowsInvalidOperationException()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);
        logger.StartGraphCapture("first_capture");

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => logger.StartGraphCapture("second_capture"));
    }

    [Fact]
    public void StopGraphCapture_WithActiveCapture_StopsCapture()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);
        logger.StartGraphCapture("captured_graph");

        // Act
        logger.StopGraphCapture();

        // Assert
        // Should not throw an exception
    }

    [Fact]
    public void StopGraphCapture_WithoutActiveCapture_ThrowsInvalidOperationException()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => logger.StopGraphCapture());
    }

    [Fact]
    public void GetCapturedGraph_AfterStartCapture_ReturnsGraph()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);
        logger.StartGraphCapture("captured_graph");

        // Act
        var graph = logger.GetCapturedGraph();

        // Assert
        Assert.NotNull(graph);
        Assert.Equal("captured_graph", graph!.Name);
    }

    [Fact]
    public void GetCapturedGraph_WithoutCapture_ReturnsNull()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);

        // Act
        var graph = logger.GetCapturedGraph();

        // Assert
        Assert.Null(graph);
    }

    [Fact]
    public void AnalyzeGraph_WithLoggedGraph_ReturnsAnalysis()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);
        var graph = new ComputationalGraph("test_graph", 0);
        var node = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        graph.AddNode(node);
        logger.LogGraph(graph);

        // Get the graph ID
        var graphId = logger.GetAllGraphs().First().Key;

        // Act
        var analysis = logger.AnalyzeGraph(graphId);

        // Assert
        Assert.NotNull(analysis);
        Assert.Equal(0, analysis.TotalParameters);
        Assert.Equal(1, analysis.TotalOperations);
    }

    [Fact]
    public void AnalyzeGraph_WithInvalidGraphId_ThrowsArgumentException()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => logger.AnalyzeGraph("invalid_id"));
    }

    [Fact]
    public void AnalyzeAllGraphs_WithLoggedGraphs_ReturnsAnalyses()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);
        var graph1 = new ComputationalGraph("graph1", 0);
        var graph2 = new ComputationalGraph("graph2", 1);
        var node1 = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        var node2 = new GraphNode("node2", "Node 2", NodeType.Operation, "ReLU");
        graph1.AddNode(node1);
        graph2.AddNode(node2);
        logger.LogGraph(graph1);
        logger.LogGraph(graph2);

        // Act
        var analyses = logger.AnalyzeAllGraphs();

        // Assert
        Assert.Equal(2, analyses.Count);
    }

    [Fact]
    public void GetAllGraphs_WithLoggedGraphs_ReturnsAllGraphs()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);
        var graph = new ComputationalGraph("test_graph", 0);
        var node = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        graph.AddNode(node);
        logger.LogGraph(graph);

        // Act
        var graphs = logger.GetAllGraphs();

        // Assert
        Assert.Single(graphs);
    }

    [Fact]
    public void ClearGraphs_WithLoggedGraphs_ClearsAllGraphs()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);
        var graph = new ComputationalGraph("test_graph", 0);
        var node = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        graph.AddNode(node);
        logger.LogGraph(graph);

        // Act
        logger.ClearGraphs();

        // Assert
        Assert.Empty(logger.GetAllGraphs());
    }

    [Fact]
    public void Dispose_WithLogger_DisposesResources()
    {
        // Arrange
        var storage = new MockStorageBackend();
        var logger = new GraphLogger(storage);
        var graph = new ComputationalGraph("test_graph", 0);
        var node = new GraphNode("node1", "Node 1", NodeType.Operation, "Conv2D");
        graph.AddNode(node);
        logger.LogGraph(graph);

        // Act
        logger.Dispose();

        // Assert
        Assert.Empty(logger.GetAllGraphs());
    }
}
