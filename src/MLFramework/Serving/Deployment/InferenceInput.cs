namespace MLFramework.Serving.Deployment;

/// <summary>
/// Represents input data for a model inference request
/// </summary>
public class InferenceInput
{
    /// <summary>
    /// The raw input data (e.g., text, image bytes, tensor data)
    /// </summary>
    public object Data { get; set; }

    /// <summary>
    /// Optional metadata about the input
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Optional request ID for tracking
    /// </summary>
    public string? RequestId { get; set; }

    public InferenceInput(object data)
    {
        Data = data;
    }

    public InferenceInput(object data, Dictionary<string, object> metadata)
    {
        Data = data;
        Metadata = metadata;
    }
}
