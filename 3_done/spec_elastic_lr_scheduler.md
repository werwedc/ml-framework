# Spec: AdaptiveLearningRateScheduler Implementation

## Overview
Implement the AdaptiveLearningRateScheduler class which adjusts the learning rate when the number of workers changes in an elastic training cluster. It supports multiple adaptation strategies (Linear, SquareRoot, None) and provides smooth transitions between learning rates.

## Deliverables

**File:** `src/MachineLearning/Distributed/Scheduling/AdaptiveLearningRateScheduler.cs`
```csharp
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
```

**File:** `src/MachineLearning/Distributed/Scheduling/LearningRateTransitionManager.cs`
```csharp
namespace MachineLearning.Distributed.Scheduling;

/// <summary>
/// Manages smooth learning rate transitions during rescaling events
/// </summary>
public class LearningRateTransitionManager
{
    private readonly AdaptiveLearningRateScheduler _scheduler;
    private readonly int _transitionSteps;
    private int _currentStep;
    private float? _oldLearningRate;
    private float? _newLearningRate;
    private bool _isTransitioning;

    /// <summary>
    /// Indicates if a transition is currently in progress
    /// </summary>
    public bool IsTransitioning => _isTransitioning;

    /// <summary>
    /// Gets the current transition progress (0.0 to 1.0)
    /// </summary>
    public float TransitionProgress => _isTransitioning ? _currentStep / (float)_transitionSteps : 0.0f;

    public LearningRateTransitionManager(
        AdaptiveLearningRateScheduler scheduler,
        int transitionSteps = 100)
    {
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));

        if (transitionSteps <= 0)
            throw new ArgumentException("Transition steps must be positive", nameof(transitionSteps));

        _transitionSteps = transitionSteps;
    }

    /// <summary>
    /// Start a learning rate transition
    /// </summary>
    public void StartTransition(int oldWorkerCount, int newWorkerCount, float currentLR)
    {
        _oldLearningRate = currentLR;
        _newLearningRate = _scheduler.AdaptLearningRate(oldWorkerCount, newWorkerCount, currentLR);
        _currentStep = 0;
        _isTransitioning = true;
    }

    /// <summary>
    /// Get the learning rate for the current step
    /// </summary>
    public float GetCurrentLearningRate()
    {
        if (!_isTransitioning || _oldLearningRate == null || _newLearningRate == null)
        {
            return _scheduler.CurrentLearningRate;
        }

        var lr = _scheduler.TransitionLearningRate(
            _oldLearningRate.Value,
            _newLearningRate.Value,
            _transitionSteps
        ).ElementAt(_currentStep);

        _currentStep++;

        if (_currentStep >= _transitionSteps)
        {
            _isTransitioning = false;
            _oldLearningRate = null;
        }

        return lr;
    }

    /// <summary>
    /// Skip the remaining transition and jump to target learning rate
    /// </summary>
    public void CompleteTransition()
    {
        if (_isTransitioning && _newLearningRate.HasValue)
        {
            _currentStep = _transitionSteps;
            _isTransitioning = false;
            _oldLearningRate = null;
        }
    }

    /// <summary>
    /// Reset the transition manager
    /// </summary>
    public void Reset()
    {
        _isTransitioning = false;
        _currentStep = 0;
        _oldLearningRate = null;
        _newLearningRate = null;
    }
}
```

## Implementation Notes

1. Adaptation Strategies:
   - Linear: Direct proportional scaling (most common)
   - SquareRoot: More conservative scaling for stability
   - None: Keep learning rate constant, only throughput changes

2. Smooth Transitions: Gradual change over configurable number of steps to avoid training instability

3. State Management: Tracks current worker count and learning rate for proper adaptation

4. Transition Manager: Separate class to manage multi-step transitions with progress tracking

## Dependencies
- AdaptationStrategy enum from spec_elastic_config_models.md
- System.Linq for ElementAt

## Estimated Effort
~40 minutes

## Success Criteria
- Linear scaling correctly multiplies LR by worker count ratio
- Square root scaling correctly uses square root transformation
- None strategy keeps LR constant
- Transition sequence generates correct interpolated values
- Transition manager correctly tracks progress
- Edge cases are handled (zero steps, negative counts, etc.)
- Reset functionality works correctly
