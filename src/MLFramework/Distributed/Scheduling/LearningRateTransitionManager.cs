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
