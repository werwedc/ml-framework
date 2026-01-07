namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Represents a single inference request.
/// </summary>
public class Request
{
    /// <summary>
    /// Unique identifier for the request.
    /// </summary>
    public RequestId Id { get; }

    /// <summary>
    /// Input text for generation.
    /// </summary>
    public string Prompt { get; }

    /// <summary>
    /// Maximum tokens to generate.
    /// </summary>
    public int MaxTokens { get; }

    /// <summary>
    /// Cancellation token for the request.
    /// </summary>
    public CancellationToken CancellationToken { get; }

    /// <summary>
    /// Task completion source for async completion.
    /// </summary>
    public TaskCompletionSource<string> CompletionSource { get; }

    /// <summary>
    /// Time when the request was enqueued.
    /// </summary>
    public DateTime EnqueuedTime { get; }

    /// <summary>
    /// Priority level for scheduling.
    /// </summary>
    public Priority Priority { get; }

    /// <summary>
    /// Count of tokens generated so far.
    /// </summary>
    public int GeneratedTokens { get; set; }

    /// <summary>
    /// Completion flag.
    /// </summary>
    public bool IsCompleted { get; set; }

    /// <summary>
    /// Accumulated token IDs.
    /// </summary>
    public List<int> GeneratedTokenIds { get; }

    /// <summary>
    /// Creates a new request with the specified parameters.
    /// </summary>
    public Request(
        RequestId id,
        string prompt,
        int maxTokens,
        CancellationToken token,
        Priority priority = Priority.Normal)
    {
        Id = id;
        Prompt = prompt;
        MaxTokens = maxTokens;
        CancellationToken = token;
        CompletionSource = new TaskCompletionSource<string>();
        EnqueuedTime = DateTime.UtcNow;
        Priority = priority;
        GeneratedTokens = 0;
        IsCompleted = false;
        GeneratedTokenIds = new List<int>();
    }
}
