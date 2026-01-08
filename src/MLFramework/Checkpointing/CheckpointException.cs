namespace MachineLearning.Checkpointing;

/// <summary>
/// Types of checkpoint exceptions
/// </summary>
public enum ExceptionType
{
    /// <summary>
    /// General checkpoint error
    /// </summary>
    General,

    /// <summary>
    /// Queue is full
    /// </summary>
    QueueFull,

    /// <summary>
    /// Checkpoint validation failed
    /// </summary>
    ValidationFailed,

    /// <summary>
    /// Storage operation failed
    /// </summary>
    StorageError,

    /// <summary>
    /// Corrupted checkpoint
    /// </summary>
    Corrupted,

    /// <summary>
    /// Timeout waiting for operation
    /// </summary>
    Timeout
}

/// <summary>
/// Base exception for checkpoint-related errors
/// </summary>
public class CheckpointException : Exception
{
    /// <summary>
    /// Gets the type of checkpoint exception
    /// </summary>
    public ExceptionType ExceptionType { get; }

    /// <summary>
    /// Gets the checkpoint ID associated with this exception, if available
    /// </summary>
    public string? CheckpointId { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="CheckpointException"/> class.
    /// </summary>
    /// <param name="message">The error message.</param>
    public CheckpointException(string message)
        : this(message, ExceptionType.General)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CheckpointException"/> class
    /// with exception type.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="exceptionType">The type of exception.</param>
    public CheckpointException(string message, ExceptionType exceptionType)
        : base(message)
    {
        ExceptionType = exceptionType;
        CheckpointId = null;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CheckpointException"/> class
    /// with checkpoint ID.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="exceptionType">The type of exception.</param>
    /// <param name="checkpointId">The checkpoint ID.</param>
    public CheckpointException(string message, ExceptionType exceptionType, string? checkpointId)
        : base(message)
    {
        ExceptionType = exceptionType;
        CheckpointId = checkpointId;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CheckpointException"/> class
    /// with a reference to the inner exception.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="innerException">The inner exception.</param>
    public CheckpointException(string message, Exception innerException)
        : base(message, innerException)
    {
        ExceptionType = ExceptionType.General;
        CheckpointId = null;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CheckpointException"/> class
    /// with a reference to the inner exception and exception type.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="exceptionType">The type of exception.</param>
    /// <param name="innerException">The inner exception.</param>
    public CheckpointException(string message, ExceptionType exceptionType, Exception innerException)
        : base(message, innerException)
    {
        ExceptionType = exceptionType;
        CheckpointId = null;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CheckpointException"/> class
    /// with a reference to the inner exception, exception type, and checkpoint ID.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="exceptionType">The type of exception.</param>
    /// <param name="checkpointId">The checkpoint ID.</param>
    /// <param name="innerException">The inner exception.</param>
    public CheckpointException(string message, ExceptionType exceptionType, string? checkpointId, Exception innerException)
        : base(message, innerException)
    {
        ExceptionType = exceptionType;
        CheckpointId = checkpointId;
    }
}
