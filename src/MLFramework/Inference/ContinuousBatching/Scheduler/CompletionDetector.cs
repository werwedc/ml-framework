namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Detects when a request has completed generation based on various conditions.
/// </summary>
public class CompletionDetector
{
    private readonly CompletionConfiguration _config;
    private readonly ITokenizer _tokenizer;

    /// <summary>
    /// Creates a new completion detector.
    /// </summary>
    /// <param name="config">Completion configuration.</param>
    /// <param name="tokenizer">Tokenizer for text operations.</param>
    public CompletionDetector(
        CompletionConfiguration config,
        ITokenizer tokenizer)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
    }

    /// <summary>
    /// Checks if a request has completed.
    /// </summary>
    /// <param name="request">The request to check.</param>
    /// <returns>Completion status and reason.</returns>
    public (bool IsCompleted, CompletionReason Reason) CheckCompletion(
        Request request)
    {
        // Check cancellation first (highest priority)
        if (CheckCancellation(request))
        {
            return (true, CompletionReason.Cancelled);
        }

        // Check max tokens
        if (CheckMaxTokens(request))
        {
            return (true, CompletionReason.MaxTokensReached);
        }

        // Check EOS token
        if (CheckEosToken(request))
        {
            return (true, CompletionReason.EosTokenReached);
        }

        // Check stop strings
        if (CheckStopString(request))
        {
            return (true, CompletionReason.StopString);
        }

        // Check length constraint
        if (CheckLengthConstraint(request))
        {
            return (true, CompletionReason.LengthReached);
        }

        return (false, CompletionReason.EosTokenReached); // Default reason
    }

    /// <summary>
    /// Checks if the last generated token is the EOS token.
    /// </summary>
    private bool CheckEosToken(Request request)
    {
        if (request.GeneratedTokenIds.Count == 0)
            return false;

        int lastToken = request.GeneratedTokenIds[^1];
        return lastToken == _config.EosTokenId;
    }

    /// <summary>
    /// Checks if max tokens limit has been reached.
    /// </summary>
    private bool CheckMaxTokens(Request request)
    {
        int maxTokens = request.MaxTokens > 0
            ? request.MaxTokens
            : _config.DefaultMaxTokens;

        return request.GeneratedTokens >= maxTokens;
    }

    /// <summary>
    /// Checks if any stop string has been encountered.
    /// </summary>
    private bool CheckStopString(Request request)
    {
        if (_config.StopStrings == null || _config.StopStrings.Count == 0)
            return false;

        if (request.GeneratedTokenIds.Count == 0)
            return false;

        // Decode recent tokens to check for stop strings
        string generatedText = _tokenizer.Decode(
            request.GeneratedTokenIds.ToArray()
        );

        foreach (var stopString in _config.StopStrings)
        {
            if (generatedText.Contains(stopString))
            {
                // Note: In production, need to update GeneratedTokenIds to remove stop string tokens
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Checks if the response length constraint has been reached.
    /// </summary>
    private bool CheckLengthConstraint(Request request)
    {
        if (_config.MaxResponseLength == null)
            return false;

        if (request.GeneratedTokenIds.Count == 0)
            return false;

        string generatedText = _tokenizer.Decode(
            request.GeneratedTokenIds.ToArray()
        );

        return generatedText.Length >= _config.MaxResponseLength.Value;
    }

    /// <summary>
    /// Checks if the request has been cancelled.
    /// </summary>
    private bool CheckCancellation(Request request)
    {
        return request.CancellationToken.IsCancellationRequested;
    }
}
