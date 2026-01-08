namespace MLFramework.Data;

/// <summary>
/// Interface for logging errors and messages from the data loading pipeline.
/// Allows for flexible logging implementations (console, file, etc.).
/// </summary>
public interface IErrorLogger
{
    /// <summary>
    /// Logs a worker error with full details.
    /// </summary>
    /// <param name="error">The error to log.</param>
    void LogError(WorkerError error);

    /// <summary>
    /// Logs a warning message.
    /// </summary>
    /// <param name="message">The warning message to log.</param>
    void LogWarning(string message);

    /// <summary>
    /// Logs an informational message.
    /// </summary>
    /// <param name="message">The informational message to log.</param>
    void LogInfo(string message);

    /// <summary>
    /// Logs a debug message (optional, may not be implemented in all loggers).
    /// </summary>
    /// <param name="message">The debug message to log.</param>
    void LogDebug(string message);
}
