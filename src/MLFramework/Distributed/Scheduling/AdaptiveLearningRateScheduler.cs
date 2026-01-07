namespace MachineLearning.Distributed.Scheduling;

using MachineLearning.Distributed.Enums;

/// <summary>
/// Adjusts learning rate based on cluster size changes
/// </summary>
public class AdaptiveLearningRateScheduler
{
    private readonly AdaptationStrategy _strategy;
    private readonly int _initialWorkerCount;
    private readonly float _initialLearningRate;
    private int _currentWorkerCount;
    private float _currentLearningRate;

    /// <summary>
    /// Gets the current learning rate
    /// </summary>
    public float CurrentLearningRate => _currentLearningRate;

    /// <summary>
    /// Gets the current worker count
    /// </summary>
    public int CurrentWorkerCount => _currentWorkerCount;

    public AdaptiveLearningRateScheduler(
        AdaptationStrategy strategy,
        int initialWorkerCount,
        float initialLearningRate)
    {
        if (initialWorkerCount <= 0)
            throw new ArgumentException("Initial worker count must be positive", nameof(initialWorkerCount));

        if (initialLearningRate <= 0)
            throw new ArgumentException("Initial learning rate must be positive", nameof(initialLearningRate));

        _strategy = strategy;
        _initialWorkerCount = initialWorkerCount;
        _initialLearningRate = initialLearningRate;
        _currentWorkerCount = initialWorkerCount;
        _currentLearningRate = initialLearningRate;
    }

    /// <summary>
    /// Calculate the new learning rate based on worker count change
    /// </summary>
    public float AdaptLearningRate(int oldWorkerCount, int newWorkerCount, float currentLR)
    {
        if (newWorkerCount <= 0)
            throw new ArgumentException("New worker count must be positive", nameof(newWorkerCount));

        if (oldWorkerCount <= 0)
            throw new ArgumentException("Old worker count must be positive", nameof(oldWorkerCount));

        float newLR;

        switch (_strategy)
        {
            case AdaptationStrategy.Linear:
                newLR = AdaptLinear(oldWorkerCount, newWorkerCount, currentLR);
                break;

            case AdaptationStrategy.SquareRoot:
                newLR = AdaptSquareRoot(oldWorkerCount, newWorkerCount, currentLR);
                break;

            case AdaptationStrategy.None:
                newLR = currentLR;
                break;

            default:
                throw new ArgumentException($"Unknown adaptation strategy: {_strategy}");
        }

        _currentWorkerCount = newWorkerCount;
        _currentLearningRate = newLR;

        return newLR;
    }

    /// <summary>
    /// Get the target learning rate for a given worker count without updating state
    /// </summary>
    public float GetTargetLearningRate(int workerCount)
    {
        return _strategy switch
        {
            AdaptationStrategy.Linear => _initialLearningRate * workerCount / _initialWorkerCount,
            AdaptationStrategy.SquareRoot => _initialLearningRate * (float)Math.Sqrt(workerCount) / (float)Math.Sqrt(_initialWorkerCount),
            AdaptationStrategy.None => _initialLearningRate,
            _ => throw new ArgumentException($"Unknown adaptation strategy: {_strategy}")
        };
    }

    /// <summary>
    /// Smoothly transition learning rate from old to new value over specified steps
    /// </summary>
    public IEnumerable<float> TransitionLearningRate(float oldLR, float newLR, int steps)
    {
        if (steps <= 0)
            throw new ArgumentException("Number of transition steps must be positive", nameof(steps));

        if (steps == 1)
        {
            yield return newLR;
            yield break;
        }

        for (int step = 0; step < steps; step++)
        {
            float t = step / (float)(steps - 1);
            float lr = oldLR + t * (newLR - oldLR);
            yield return lr;
        }
    }

    /// <summary>
    /// Reset the scheduler to initial state
    /// </summary>
    public void Reset()
    {
        _currentWorkerCount = _initialWorkerCount;
        _currentLearningRate = _initialLearningRate;
    }

    private float AdaptLinear(int oldCount, int newCount, float currentLR)
    {
        // Scale learning rate proportionally to worker count
        // LR_new = LR_old * (newCount / oldCount)
        return currentLR * newCount / oldCount;
    }

    private float AdaptSquareRoot(int oldCount, int newCount, float currentLR)
    {
        // Scale learning rate with square root of worker count for more stability
        // LR_new = LR_old * sqrt(newCount / oldCount)
        return currentLR * (float)Math.Sqrt((double)newCount / oldCount);
    }
}
