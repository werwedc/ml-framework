namespace MLFramework.Schedulers;

/// <summary>
/// Marker interface for schedulers that step on each batch/iteration.
/// </summary>
public interface IStepScheduler : ILearningRateScheduler
{
    // Marker interface - no additional methods needed
}
