namespace MLFramework.Distributed;

/// <summary>
/// Exception thrown when distributed training fails.
/// </summary>
public class DistributedTrainingException : Exception
{
    /// <summary>
    /// Gets the rank of the process that failed.
    /// </summary>
    public int Rank { get; }

    /// <summary>
    /// Gets the exit code of the failed process.
    /// </summary>
    public int ExitCode { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="DistributedTrainingException"/> class.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="rank">The rank of the failed process (-1 if unknown).</param>
    /// <param name="exitCode">The exit code of the failed process (-1 if unknown).</param>
    public DistributedTrainingException(string message, int rank = -1, int exitCode = -1)
        : base(message)
    {
        Rank = rank;
        ExitCode = exitCode;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DistributedTrainingException"/> class
    /// with a reference to the inner exception.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="innerException">The inner exception.</param>
    /// <param name="rank">The rank of the failed process (-1 if unknown).</param>
    /// <param name="exitCode">The exit code of the failed process (-1 if unknown).</param>
    public DistributedTrainingException(
        string message,
        Exception innerException,
        int rank = -1,
        int exitCode = -1)
        : base(message, innerException)
    {
        Rank = rank;
        ExitCode = exitCode;
    }
}
