namespace MachineLearning.Checkpointing;

/// <summary>
/// Exception thrown when the checkpoint queue is full and cannot accept more tasks
/// </summary>
public class CheckpointQueueFullException : CheckpointException
{
    /// <summary>
    /// Gets the current number of tasks in the queue
    /// </summary>
    public int CurrentQueueSize { get; }

    /// <summary>
    /// Gets the maximum queue size
    /// </summary>
    public int MaxQueueSize { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="CheckpointQueueFullException"/> class.
    /// </summary>
    /// <param name="message">The error message.</param>
    public CheckpointQueueFullException(string message)
        : base(message, ExceptionType.QueueFull)
    {
        CurrentQueueSize = 0;
        MaxQueueSize = 0;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CheckpointQueueFullException"/> class
    /// with queue size information.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="currentQueueSize">The current number of tasks in the queue.</param>
    /// <param name="maxQueueSize">The maximum queue size.</param>
    public CheckpointQueueFullException(string message, int currentQueueSize, int maxQueueSize)
        : base(message, ExceptionType.QueueFull)
    {
        CurrentQueueSize = currentQueueSize;
        MaxQueueSize = maxQueueSize;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CheckpointQueueFullException"/> class
    /// with queue size information and checkpoint ID.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="checkpointId">The checkpoint ID that was rejected.</param>
    /// <param name="currentQueueSize">The current number of tasks in the queue.</param>
    /// <param name="maxQueueSize">The maximum queue size.</param>
    public CheckpointQueueFullException(string message, string checkpointId, int currentQueueSize, int maxQueueSize)
        : base(message, ExceptionType.QueueFull, checkpointId)
    {
        CurrentQueueSize = currentQueueSize;
        MaxQueueSize = maxQueueSize;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CheckpointQueueFullException"/> class
    /// with a reference to the inner exception.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="innerException">The inner exception.</param>
    public CheckpointQueueFullException(string message, Exception innerException)
        : base(message, ExceptionType.QueueFull, innerException)
    {
        CurrentQueueSize = 0;
        MaxQueueSize = 0;
    }
}
