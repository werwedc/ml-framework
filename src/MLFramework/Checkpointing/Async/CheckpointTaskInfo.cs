namespace MachineLearning.Checkpointing.Async;

/// <summary>
/// Information about a checkpoint task (read-only view)
/// </summary>
public class CheckpointTaskInfo
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
    /// Gets or sets the current status of the task
    /// </summary>
    public CheckpointTaskStatus Status { get; set; }

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
    /// Gets the duration of the task execution
    /// </summary>
    public TimeSpan? Duration => CompletedAt.HasValue && StartedAt.HasValue
        ? CompletedAt.Value - StartedAt.Value
        : null;

    /// <summary>
    /// Gets the time elapsed since the task was queued
    /// </summary>
    public TimeSpan TimeSinceQueued => DateTime.UtcNow - QueuedAt;

    /// <summary>
    /// Gets whether the task is still active (queued or running)
    /// </summary>
    public bool IsActive => Status == CheckpointTaskStatus.Queued || Status == CheckpointTaskStatus.Running;

    /// <summary>
    /// Gets whether the task completed successfully
    /// </summary>
    public bool IsCompletedSuccessfully => Status == CheckpointTaskStatus.Completed;

    /// <summary>
    /// Gets whether the task failed
    /// </summary>
    public bool IsFailed => Status == CheckpointTaskStatus.Failed;
}
