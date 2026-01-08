namespace MachineLearning.Checkpointing.Async;

/// <summary>
/// Result of a checkpoint task
/// </summary>
public class CheckpointTaskResult
{
    /// <summary>
    /// Gets or sets whether the task was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets the path where the checkpoint was saved
    /// </summary>
    public string? CheckpointPath { get; set; }

    /// <summary>
    /// Gets or sets the error message if the task failed
    /// </summary>
    public string? Error { get; set; }

    /// <summary>
    /// Gets or sets the duration of the task execution
    /// </summary>
    public TimeSpan Duration { get; set; }

    /// <summary>
    /// Gets or sets the number of bytes written
    /// </summary>
    public long BytesWritten { get; set; }

    /// <summary>
    /// Gets or sets the exception if the task failed
    /// </summary>
    public Exception? Exception { get; set; }

    /// <summary>
    /// Creates a successful checkpoint task result
    /// </summary>
    /// <param name="checkpointPath">The checkpoint path.</param>
    /// <param name="duration">The duration of the operation.</param>
    /// <param name="bytesWritten">The number of bytes written.</param>
    /// <returns>A successful checkpoint task result.</returns>
    public static CheckpointTaskResult CreateSuccess(string checkpointPath, TimeSpan duration, long bytesWritten = 0)
    {
        return new CheckpointTaskResult
        {
            Success = true,
            CheckpointPath = checkpointPath,
            Duration = duration,
            BytesWritten = bytesWritten
        };
    }

    /// <summary>
    /// Creates a failed checkpoint task result
    /// </summary>
    /// <param name="error">The error message.</param>
    /// <param name="exception">The exception that caused the failure.</param>
    /// <returns>A failed checkpoint task result.</returns>
    public static CheckpointTaskResult CreateFailure(string error, Exception? exception = null)
    {
        return new CheckpointTaskResult
        {
            Success = false,
            Error = error,
            Exception = exception
        };
    }
}
