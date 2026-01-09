using System;
using System.Collections.Generic;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Captures operation context when errors occur to provide detailed diagnostic information.
/// </summary>
public class OperationExecutionContext
{
    /// <summary>
    /// Gets or sets the name of the layer/module where the operation is executing.
    /// </summary>
    public string LayerName { get; set; }

    /// <summary>
    /// Gets or sets the type of operation being performed.
    /// </summary>
    public OperationType OperationType { get; set; }

    /// <summary>
    /// Gets or sets the input tensors for the operation.
    /// </summary>
    public global::RitterFramework.Core.Tensor.Tensor[] InputTensors { get; set; }

    /// <summary>
    /// Gets or sets additional parameters for the operation (e.g., stride, padding).
    /// </summary>
    public IDictionary<string, object> OperationParameters { get; set; }

    /// <summary>
    /// Gets or sets the name of the previous layer in the computation graph.
    /// </summary>
    public string PreviousLayerName { get; set; }

    /// <summary>
    /// Gets or sets the output tensor from the previous layer.
    /// </summary>
    public global::RitterFramework.Core.Tensor.Tensor PreviousLayerOutput { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when this context was captured.
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Creates a new instance of OperationExecutionContext.
    /// </summary>
    public OperationExecutionContext()
    {
        OperationParameters = new Dictionary<string, object>();
        InputTensors = Array.Empty<global::RitterFramework.Core.Tensor.Tensor>();
        Timestamp = DateTime.UtcNow;
    }

    /// <summary>
    /// Gets the input shapes as arrays.
    /// </summary>
    /// <returns>Array of input shapes.</returns>
    public long[][] GetInputShapes()
    {
        if (InputTensors == null || InputTensors.Length == 0)
        {
            return Array.Empty<long[]>();
        }

        var shapes = new long[InputTensors.Length][];
        for (int i = 0; i < InputTensors.Length; i++)
        {
            shapes[i] = InputTensors[i].Shape.Select(d => (long)d).ToArray();
        }
        return shapes;
    }

    /// <summary>
    /// Gets the previous layer shape as an array.
    /// </summary>
    /// <returns>Previous layer shape, or null if not available.</returns>
    public long[]? GetPreviousLayerShape()
    {
        return PreviousLayerOutput?.Shape.Select(d => (long)d).ToArray();
    }
}
