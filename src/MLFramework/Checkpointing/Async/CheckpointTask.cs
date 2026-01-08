namespace MachineLearning.Checkpointing.Async;

/// <summary>
/// Type of checkpoint task
/// </summary>
public enum CheckpointTaskType
{
    /// <summary>
    /// Save checkpoint task
    /// </summary>
    Save,

    /// <summary>
    /// Load checkpoint task
    /// </summary>
    Load
}

/// <summary>
/// Status of a checkpoint task
/// </summary>
public enum CheckpointTaskStatus
{
    /// <summary>
    /// Task is queued and waiting to be processed
    /// </summary>
    Queued,

    /// <summary>
    /// Task is currently being processed
    /// </summary>
    Running,

    /// <summary>
    /// Task completed successfully
    /// </summary>
    Completed,

    /// <summary>
    /// Task failed with an error
    /// </summary>
    Failed,

    /// <summary>
    /// Task was cancelled before processing
    /// </summary>
    Cancelled,

    /// <summary>
    /// Task was rejected (e.g., queue was full)
    /// </summary>
    Rejected
}

/// <summary>
/// Represents a checkpoint task in the async queue
/// </summary>
public class CheckpointTask
{
    /// <summary>
    /// Gets or sets the unique ID of the checkpoint task
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the type of checkpoint task
    /// </summary>
    public CheckpointTaskType Type { get; set; }

    /// <summary>
    /// Gets or sets the model state to checkpoint
    /// </summary>
    public IStateful? Model { get; set; }

    /// <summary>
    /// Gets or sets the optimizer state to checkpoint
    /// </summary>
    public IStateful? Optimizer { get; set; }

    /// <summary>
    /// Gets or sets the save options for this task
    /// </summary>
    public SaveOptions Options { get; set; } = null!;

    /// <summary>
    /// Gets or sets the time when the task was queued
    /// </summary>
    public DateTime QueuedAt { get; set; }

    /// <summary>
    /// Gets or sets the time when the task started processing
    /// </summary>
    public DateTime? StartedAt { get; set; }

    /// <summary>
    /// Gets or sets the time when the task completed
    /// </summary>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Gets or sets the current status of the task
    /// </summary>
    public CheckpointTaskStatus Status { get; set; }

    /// <summary>
    /// Gets the task completion source
    /// </summary>
    public TaskCompletionSource<CheckpointTaskResult> CompletionSource { get; set; } = new();

    /// <summary>
    /// Gets the task completion task
    /// </summary>
    public Task<CheckpointTaskResult> CompletionTask => CompletionSource.Task;

    /// <summary>
    /// Gets or sets the result of the task (set when completed)
    /// </summary>
    public CheckpointTaskResult? Result { get; set; }

    /// <summary>
    /// Gets the duration of the task execution
    /// </summary>
    public TimeSpan? Duration => StartedAt.HasValue && CompletedAt.HasValue
        ? CompletedAt.Value - StartedAt.Value
        : null;
}
