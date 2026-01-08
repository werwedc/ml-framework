namespace MLFramework.Schedulers;

/// <summary>
/// Marker interface for schedulers that step on each epoch.
/// </summary>
public interface IEpochScheduler : ILearningRateScheduler
{
    /// <summary>
    /// Advances the scheduler by one epoch.
    /// Called at the end of each epoch.
    /// </summary>
    void StepEpoch();
}
