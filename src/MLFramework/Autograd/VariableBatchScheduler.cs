namespace MLFramework.Autograd;

/// <summary>
/// Manages batch size scheduling for gradient accumulation with variable batch sizes.
/// Allows pre-defining batch sizes for each step and tracking progress.
/// </summary>
public class VariableBatchScheduler
{
    private readonly List<int> _batchSchedule;
    private int _currentStep;

    /// <summary>
    /// Gets the list of batch sizes scheduled for each step.
    /// </summary>
    public IReadOnlyList<int> BatchSchedule => _batchSchedule.AsReadOnly();

    /// <summary>
    /// Gets the current step index (0-based).
    /// </summary>
    public int CurrentStep => _currentStep;

    /// <summary>
    /// Gets the total number of steps in the schedule.
    /// </summary>
    public int TotalSteps => _batchSchedule.Count;

    /// <summary>
    /// Gets the effective batch size (sum of all scheduled batch sizes).
    /// </summary>
    public int EffectiveBatchSize => _batchSchedule.Sum();

    /// <summary>
    /// Initializes a new instance of the VariableBatchScheduler class with a predefined schedule.
    /// </summary>
    /// <param name="batchSchedule">List of batch sizes for each step.</param>
    /// <exception cref="ArgumentNullException">Thrown when batchSchedule is null.</exception>
    /// <exception cref="ArgumentException">Thrown when batchSchedule is empty or contains invalid values.</exception>
    public VariableBatchScheduler(List<int> batchSchedule)
    {
        if (batchSchedule == null)
            throw new ArgumentNullException(nameof(batchSchedule));

        if (batchSchedule.Count == 0)
            throw new ArgumentException("Batch schedule cannot be empty", nameof(batchSchedule));

        foreach (var batchSize in batchSchedule)
        {
            if (batchSize < 1)
                throw new ArgumentException($"Batch size {batchSize} is invalid (must be at least 1)", nameof(batchSchedule));
        }

        _batchSchedule = new List<int>(batchSchedule);
        _currentStep = 0;
    }

    /// <summary>
    /// Initializes a new instance of the VariableBatchScheduler class from an array.
    /// </summary>
    /// <param name="batchSizes">Array of batch sizes for each step.</param>
    public VariableBatchScheduler(params int[] batchSizes) : this(new List<int>(batchSizes))
    {
    }

    /// <summary>
    /// Gets the batch size for the current step.
    /// </summary>
    /// <returns>The batch size for the current step.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the schedule is complete.</exception>
    public int GetCurrentBatchSize()
    {
        if (_currentStep >= _batchSchedule.Count)
            throw new InvalidOperationException("Schedule is complete. No more steps available.");

        return _batchSchedule[_currentStep];
    }

    /// <summary>
    /// Gets the number of remaining steps in the schedule.
    /// </summary>
    /// <returns>The count of remaining steps.</returns>
    public int GetRemainingSteps()
    {
        return Math.Max(0, _batchSchedule.Count - _currentStep);
    }

    /// <summary>
    /// Gets the effective batch size over a sliding window of steps.
    /// </summary>
    /// <param name="windowSize">The size of the window to average over.</param>
    /// <returns>The average effective batch size over the window.</returns>
    /// <exception cref="ArgumentException">Thrown when windowSize is less than 1.</exception>
    public int GetEffectiveBatchSize(int windowSize)
    {
        if (windowSize < 1)
            throw new ArgumentException("Window size must be at least 1", nameof(windowSize));

        int startIdx = Math.Max(0, _currentStep);
        int endIdx = Math.Min(_batchSchedule.Count, startIdx + windowSize);

        if (startIdx >= _batchSchedule.Count)
            return 0;

        int sum = 0;
        for (int i = startIdx; i < endIdx; i++)
        {
            sum += _batchSchedule[i];
        }

        return sum;
    }

    /// <summary>
    /// Advances to the next step in the schedule.
    /// </summary>
    /// <returns>True if advanced successfully, false if schedule is complete.</returns>
    public bool Advance()
    {
        if (_currentStep < _batchSchedule.Count)
        {
            _currentStep++;
            return true;
        }
        return false;
    }

    /// <summary>
    /// Gets a value indicating whether the schedule is complete.
    /// </summary>
    public bool IsComplete => _currentStep >= _batchSchedule.Count;

    /// <summary>
    /// Resets the scheduler to the beginning of the schedule.
    /// </summary>
    public void Reset()
    {
        _currentStep = 0;
    }

    /// <summary>
    /// Gets the progress of the schedule as a value between 0.0 and 1.0.
    /// </summary>
    /// <returns>Progress value where 1.0 indicates completion.</returns>
    public double GetProgress()
    {
        if (_batchSchedule.Count == 0)
            return 1.0;

        return Math.Min(1.0, (double)_currentStep / _batchSchedule.Count);
    }

    /// <summary>
    /// Gets the accumulated batch size up to the current step.
    /// </summary>
    /// <returns>The sum of batch sizes from step 0 to current step.</returns>
    public int GetAccumulatedBatchSize()
    {
        int sum = 0;
        for (int i = 0; i < _currentStep && i < _batchSchedule.Count; i++)
        {
            sum += _batchSchedule[i];
        }
        return sum;
    }
}
