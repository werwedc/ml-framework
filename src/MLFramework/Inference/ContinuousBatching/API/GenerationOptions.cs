namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Generation parameters for text generation requests.
/// </summary>
public record class GenerationOptions(
    float? Temperature,                    // Sampling temperature (0-2)
    float? TopP,                          // Nucleus sampling threshold
    int? TopK,                            // Top-K sampling
    float? FrequencyPenalty,              // Repetition penalty
    float? PresencePenalty,               // Presence penalty
    List<string>? StopSequences,          // Stop sequences
    int? Seed,                            // Random seed for reproducibility
    bool? EchoPrompt,                     // Include prompt in output
    string? Grammar,                      // Structured output grammar
    Dictionary<string, object>? Metadata // Additional metadata
)
{
    /// <summary>
    /// Default generation options.
    /// </summary>
    public static readonly GenerationOptions Default = new(
        Temperature: 1.0f,
        TopP: 1.0f,
        TopK: null,
        FrequencyPenalty: 0.0f,
        PresencePenalty: 0.0f,
        StopSequences: null,
        Seed: null,
        EchoPrompt: false,
        Grammar: null,
        Metadata: null
    );
}
