namespace MLFramework.Schedulers;

/// <summary>
/// Abstract base class providing common functionality for all schedulers.
/// </summary>
public abstract class BaseScheduler : ILearningRateScheduler
{
    protected int _stepCount;
    protected int _epochCount;

    /// <summary>
    /// Gets the current step count.
    /// </summary>
    public int StepCount => _stepCount;

    /// <summary>
    /// Gets the current epoch count.
    /// </summary>
    public int EpochCount => _epochCount;

    public abstract float GetLearningRate(int step, float baseLearningRate);

    public virtual void Step()
    {
        _stepCount++;
    }

    public virtual void StepEpoch()
    {
        _epochCount++;
    }

    public virtual void Reset()
    {
        _stepCount = 0;
        _epochCount = 0;
    }

    public abstract StateDict GetState();

    public abstract void LoadState(StateDict state);
}
