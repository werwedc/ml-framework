using System.Text.Json;

namespace MachineLearning.Visualization.Events;

/// <summary>
/// Event representing a tensor operation (e.g., matrix multiplication, convolution)
/// </summary>
public class TensorOperationEvent : Event
{
    /// <summary>
    /// Name of the operation (e.g., "MatMul", "Conv2D")
    /// </summary>
    public string OperationName { get; }

    /// <summary>
    /// Shape of the input tensors
    /// </summary>
    public int[][] InputShapes { get; }

    /// <summary>
    /// Shape of the output tensor
    /// </summary>
    public int[] OutputShape { get; }

    /// <summary>
    /// Training step at which this operation occurred
    /// </summary>
    public long Step { get; }

    /// <summary>
    /// Duration of the operation in nanoseconds
    /// </summary>
    public long DurationNanoseconds { get; }

    /// <summary>
    /// Additional metadata for this operation
    /// </summary>
    public Dictionary<string, string> Metadata { get; }

    /// <summary>
    /// Creates a new tensor operation event
    /// </summary>
    /// <param name="operationName">Name of the operation</param>
    /// <param name="inputShapes">Shapes of input tensors</param>
    /// <param name="outputShape">Shape of output tensor</param>
    /// <param name="step">Training step</param>
    /// <param name="durationNanoseconds">Duration of the operation</param>
    /// <param name="metadata">Optional metadata</param>
    public TensorOperationEvent(
        string operationName,
        int[][] inputShapes,
        int[] outputShape,
        long step = -1,
        long durationNanoseconds = 0,
        Dictionary<string, string>? metadata = null)
    {
        OperationName = operationName ?? throw new ArgumentNullException(nameof(operationName));
        InputShapes = inputShapes ?? Array.Empty<int[]>();
        OutputShape = outputShape ?? Array.Empty<int>();
        Step = step;
        DurationNanoseconds = durationNanoseconds;
        Metadata = metadata ?? new Dictionary<string, string>();
    }

    /// <summary>
    /// Serializes the event to bytes
    /// </summary>
    public override byte[] Serialize()
    {
        var data = new Dictionary<string, object>
        {
            { "type", MachineLearning.Visualization.Events.EventType.TensorOperation.ToString() },
            { "operationName", OperationName },
            { "inputShapes", InputShapes },
            { "outputShape", OutputShape },
            { "step", Step },
            { "durationNanoseconds", DurationNanoseconds },
            { "timestamp", Timestamp.ToString("o") },
            { "eventId", EventId.ToString() },
            { "metadata", Metadata }
        };

        return JsonSerializer.SerializeToUtf8Bytes(data);
    }

    /// <summary>
    /// Deserializes bytes back to an event
    /// </summary>
    public override void Deserialize(byte[] data)
    {
        // Note: In a real implementation, we would update the properties from the data
        // This is a placeholder that shows the concept
        var json = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(data);
        // Implementation would parse and update properties
    }
}
