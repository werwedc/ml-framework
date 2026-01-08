using MachineLearning.Visualization.Events;
using MLFramework.Visualization.Graphs;
using System.Text.Json;

namespace MLFramework.Visualization.Events;

/// <summary>
/// Event representing a computational graph for visualization.
/// </summary>
public class ComputationalGraphEvent : Event
{
    /// <summary>
    /// Name of the graph.
    /// </summary>
    public string GraphName { get; }

    /// <summary>
    /// Step number (e.g., training step).
    /// </summary>
    public long Step { get; }

    /// <summary>
    /// Timestamp when the graph was created.
    /// </summary>
    public DateTime GraphTimestamp { get; }

    /// <summary>
    /// Number of nodes in the graph.
    /// </summary>
    public int NodeCount { get; }

    /// <summary>
    /// Number of edges in the graph.
    /// </summary>
    public int EdgeCount { get; }

    /// <summary>
    /// Depth of the graph.
    /// </summary>
    public int Depth { get; }

    /// <summary>
    /// Serialized graph data (JSON format).
    /// </summary>
    public byte[] GraphData { get; }

    /// <summary>
    /// Creates a new ComputationalGraphEvent.
    /// </summary>
    /// <param name="graph">The computational graph to log.</param>
    public ComputationalGraphEvent(ComputationalGraph graph)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));

        GraphName = graph.Name;
        Step = graph.Step;
        GraphTimestamp = graph.Timestamp;
        NodeCount = graph.NodeCount;
        EdgeCount = graph.EdgeCount;
        Depth = graph.Depth;

        // Serialize the graph to JSON
        GraphData = SerializeGraphToJson(graph);
    }

    private byte[] SerializeGraphToJson(ComputationalGraph graph)
    {
        var graphData = new
        {
            name = graph.Name,
            step = graph.Step,
            timestamp = graph.Timestamp,
            nodes = graph.Nodes.Values.Select(n => new
            {
                id = n.Id,
                name = n.Name,
                type = n.Type.ToString(),
                opType = n.OpType,
                shape = n.Shape,
                dataType = n.DataType.ToString(),
                inputs = n.InputIds,
                outputs = n.OutputIds,
                controlDependencies = n.ControlDependencies,
                attributes = n.Attributes,
                metadata = n.Metadata
            }),
            edges = graph.Edges
        };

        return JsonSerializer.SerializeToUtf8Bytes(graphData);
    }

    /// <summary>
    /// Serializes the event to bytes.
    /// </summary>
    public override byte[] Serialize()
    {
        var eventData = new
        {
            eventId = EventId,
            timestamp = Timestamp,
            eventType = EventType,
            graphName = GraphName,
            step = Step,
            graphTimestamp = GraphTimestamp,
            nodeCount = NodeCount,
            edgeCount = EdgeCount,
            depth = Depth,
            graphData = GraphData
        };

        return JsonSerializer.SerializeToUtf8Bytes(eventData);
    }

    /// <summary>
    /// Deserializes bytes back to an event.
    /// </summary>
    public override void Deserialize(byte[] data)
    {
        throw new NotImplementedException("Deserialization from bytes is not supported for this event type.");
    }

    /// <summary>
    /// Gets the graph data as a JSON string.
    /// </summary>
    public string GetGraphDataAsJson()
    {
        return System.Text.Encoding.UTF8.GetString(GraphData);
    }
}
