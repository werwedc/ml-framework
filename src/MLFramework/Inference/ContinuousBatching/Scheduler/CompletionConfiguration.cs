namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Configuration for completion detection in continuous batching.
/// </summary>
public record class CompletionConfiguration(
    int EosTokenId,                    // End-of-sequence token ID
    int DefaultMaxTokens,              // Default max tokens if not specified
    List<string>? StopStrings,         // Strings that stop generation
    int? MaxResponseLength,            // Max response length in characters
    bool EnableEarlyStopping,          // Stop if confidence low
    double EarlyStoppingThreshold      // Confidence threshold
)
{
    /// <summary>
    /// Default completion configuration.
    /// </summary>
    public static readonly CompletionConfiguration Default = new(
        EosTokenId: 2,  // Common EOS token ID
        DefaultMaxTokens: 256,
        StopStrings: null,
        MaxResponseLength: null,
        EnableEarlyStopping: false,
        EarlyStoppingThreshold: 0.01
    );
}
