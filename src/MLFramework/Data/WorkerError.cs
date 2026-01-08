namespace MLFramework.Data;

/// <summary>
/// Encapsulates information about worker errors that occur during data loading operations.
/// </summary>
public sealed class WorkerError
{
    /// <summary>
    /// Gets the unique identifier of the worker that encountered the error.
    /// </summary>
    public int WorkerId { get; }

    /// <summary>
    /// Gets the exception that caused the error.
    /// </summary>
    public Exception Exception { get; }

    /// <summary>
    /// Gets the timestamp when the error occurred.
    /// </summary>
    public DateTime Timestamp { get; }

    /// <summary>
    /// Gets additional context about where the error occurred (e.g., "WorkerLoop", "DataFetch", etc.).
    /// </summary>
    public string Context { get; }

    /// <summary>
    /// Initializes a new instance of the WorkerError class.
    /// </summary>
    /// <param name="workerId">The unique identifier of the worker that encountered the error.</param>
    /// <param name="exception">The exception that caused the error.</param>
    /// <param name="context">Additional context about where the error occurred.</param>
    /// <exception cref="ArgumentNullException">Thrown when exception is null.</exception>
    public WorkerError(int workerId, Exception exception, string context = "")
    {
        WorkerId = workerId;
        Exception = exception ?? throw new ArgumentNullException(nameof(exception));
        Timestamp = DateTime.UtcNow;
        Context = context ?? string.Empty;
    }

    /// <summary>
    /// Returns a human-readable string representation of this error.
    /// </summary>
    public override string ToString()
    {
        var contextStr = string.IsNullOrEmpty(Context) ? "" : $" in {Context}";
        return $"WorkerError {{ WorkerId: {WorkerId}, Exception: {Exception.GetType().Name}, Message: {Exception.Message}, Timestamp: {Timestamp:yyyy-MM-dd HH:mm:ss.fff} UTC{contextStr} }}";
    }

    /// <summary>
    /// Gets the full stack trace of the exception.
    /// </summary>
    public string GetStackTrace()
    {
        return Exception.StackTrace ?? "No stack trace available.";
    }

    /// <summary>
    /// Gets a formatted message including the error details.
    /// </summary>
    public string GetFormattedMessage()
    {
        var contextStr = string.IsNullOrEmpty(Context) ? "" : $" [{Context}]";
        return $"[{Timestamp:yyyy-MM-dd HH:mm:ss.fff} UTC] Worker {WorkerId}{contextStr}: {Exception.GetType().Name} - {Exception.Message}";
    }
}
