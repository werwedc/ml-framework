namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Encapsulates a generation request for batch processing.
/// </summary>
public record class GenerationRequest(
    string Prompt,
    int MaxTokens = 256,
    Priority Priority = Priority.Normal,
    GenerationOptions? Options = null,
    CancellationToken CancellationToken = default,
    Dictionary<string, object>? Metadata = null
)
{
}
