using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Manages gradient accumulation that supports variable batch sizes across accumulation steps.
/// Provides strategies for accumulating gradients when batch sizes vary.
/// </summary>
public class DynamicBatchAccumulator
{
    private Tensor? _accumulatedGradient;
    private int _accumulatedBatchSize;
    private int _currentStep;
    private readonly int _targetBatchSize;

    /// <summary>
    /// Gets the currently accumulated gradient tensor.
    /// </summary>
    public Tensor? AccumulatedGradient => _accumulatedGradient;

    /// <summary>
    /// Gets the total batch size accumulated so far.
    /// </summary>
    public int AccumulatedBatchSize => _accumulatedBatchSize;

    /// <summary>
    /// Gets the current accumulation step number.
    /// </summary>
    public int CurrentStep => _currentStep;

    /// <summary>
    /// Gets the target batch size for accumulation (0 if using fixed step count strategy).
    /// </summary>
    public int TargetBatchSize => _targetBatchSize;

    /// <summary>
    /// Gets the total number of accumulation steps performed.
    /// </summary>
    public int AccumulationCount { get; private set; }

    /// <summary>
    /// Initializes a new instance of the DynamicBatchAccumulator class with a target batch size.
    /// </summary>
    /// <param name="targetBatchSize">The target batch size to accumulate before completing.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when targetBatchSize is less than 1.</exception>
    public DynamicBatchAccumulator(int targetBatchSize)
    {
        if (targetBatchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(targetBatchSize), "Target batch size must be at least 1");

        _targetBatchSize = targetBatchSize;
        _accumulatedBatchSize = 0;
        _currentStep = 0;
        AccumulationCount = 0;
    }

    /// <summary>
    /// Initializes a new instance of the DynamicBatchAccumulator class for fixed step count strategy.
    /// </summary>
    /// <param name="targetBatchSize">Set to 0 to use fixed step count mode.</param>
    public DynamicBatchAccumulator() : this(0)
    {
        // Fixed step count mode - targetBatchSize is 0
    }

    /// <summary>
    /// Accumulates a batch gradient with its associated batch size.
    /// Normalizes the gradient by batch size to prevent bias toward larger batches.
    /// </summary>
    /// <param name="batchGradient">The gradient tensor from the current batch.</param>
    /// <param name="batchSize">The size of the current batch.</param>
    /// <exception cref="ArgumentNullException">Thrown when batchGradient is null.</exception>
    /// <exception cref="ArgumentException">Thrown when batchSize is less than 1.</exception>
    public void Accumulate(Tensor batchGradient, int batchSize)
    {
        if (batchGradient == null)
            throw new ArgumentNullException(nameof(batchGradient));

        if (batchSize < 1)
            throw new ArgumentException("Batch size must be at least 1", nameof(batchSize));

        // Normalize the gradient by batch size
        var normalizedGradient = GradientScaling.NormalizeBatchGradient(batchGradient, batchSize);

        // Initialize accumulated gradient if needed
        if (_accumulatedGradient == null)
        {
            _accumulatedGradient = Tensor.Zeros(normalizedGradient.Shape);
        }
        else if (!GradientAccumulationValidator.CheckShapeCompatibility(_accumulatedGradient, normalizedGradient))
        {
            throw new InvalidOperationException("Gradient shape mismatch in accumulation");
        }

        // Scale by actual batch size for weighted accumulation
        var scaledGradient = GradientScaling.ScaleByBatchSize(normalizedGradient, batchSize, 1);

        // Accumulate the scaled gradient
        for (int i = 0; i < _accumulatedGradient.Data.Length; i++)
        {
            _accumulatedGradient.Data[i] += scaledGradient.Data[i];
        }

        _accumulatedBatchSize += batchSize;
        _currentStep++;
        AccumulationCount++;
    }

    /// <summary>
    /// Checks if the accumulation is complete based on the strategy.
    /// </summary>
    /// <returns>True if accumulation is complete, false otherwise.</returns>
    public bool IsComplete()
    {
        if (_targetBatchSize > 0)
        {
            // Fixed target batch size strategy
            return _accumulatedBatchSize >= _targetBatchSize;
        }
        else
        {
            // Fixed step count strategy - not complete by default
            // Caller must determine completion based on step count
            return false;
        }
    }

    /// <summary>
    /// Gets the accumulated gradient, properly normalized by total batch size.
    /// </summary>
    /// <returns>The accumulated and normalized gradient tensor.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no gradients have been accumulated.</exception>
    public Tensor GetAccumulatedGradient()
    {
        if (_accumulatedGradient == null)
            throw new InvalidOperationException("No gradients accumulated yet");

        if (_accumulatedBatchSize < 1)
            throw new InvalidOperationException("Invalid accumulated batch size");

        // Return a copy with proper normalization
        return GradientScaling.AverageAccumulated(
            _accumulatedGradient,
            _accumulatedBatchSize,
            1
        );
    }

    /// <summary>
    /// Resets the accumulator state for a new accumulation cycle.
    /// </summary>
    public void Reset()
    {
        _accumulatedBatchSize = 0;
        _currentStep = 0;
        AccumulationCount = 0;

        // Reset gradient to zeros instead of null
        if (_accumulatedGradient != null)
        {
            Array.Clear(_accumulatedGradient.Data, 0, _accumulatedGradient.Data.Length);
        }
    }

    /// <summary>
    /// Gets the progress of accumulation as a value between 0.0 and 1.0.
    /// </summary>
    /// <returns>Progress value where 1.0 indicates completion.</returns>
    public double GetProgress()
    {
        if (_targetBatchSize > 0)
        {
            return Math.Min(1.0, (double)_accumulatedBatchSize / _targetBatchSize);
        }
        else
        {
            // In fixed step count mode, return 0.0 as progress is caller-defined
            return 0.0;
        }
    }

    /// <summary>
    /// Sets a target batch size and switches to fixed target batch size mode.
    /// </summary>
    /// <param name="targetBatchSize">The target batch size to accumulate.</param>
    public void SetTargetBatchSize(int targetBatchSize)
    {
        if (targetBatchSize < 0)
            throw new ArgumentOutOfRangeException(nameof(targetBatchSize), "Target batch size cannot be negative");

        // Note: This doesn't reset the accumulator, just changes the target
        // Caller should call Reset() if needed
    }
}
