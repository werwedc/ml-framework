namespace MLFramework.Serving.Deployment;

/// <summary>
/// Represents the result of a model inference operation
/// </summary>
public class InferenceResult
{
    /// <summary>
    /// The inference output data (e.g., predictions, embeddings, generated text)
    /// </summary>
    public object Data { get; set; }

    /// <summary>
    /// Whether the inference was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if inference failed
    /// </summary>
    public string? Error { get; set; }

    /// <summary>
    /// Time taken to perform the inference in milliseconds
    /// </summary>
    public long InferenceTimeMs { get; set; }

    /// <summary>
    /// Optional metadata about the result
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();

    public InferenceResult(object data, bool success = true)
    {
        Data = data;
        Success = success;
    }

    public InferenceResult(string error)
    {
        Data = null!;
        Success = false;
        Error = error;
    }

    public static InferenceResult FromError(string error)
    {
        return new InferenceResult(error);
    }
}
