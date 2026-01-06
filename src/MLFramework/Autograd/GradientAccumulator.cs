using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Manages gradient accumulation across multiple forward/backward passes.
/// Enables training with larger effective batch sizes by accumulating gradients before applying them.
/// </summary>
public class GradientAccumulator
{
    private readonly Dictionary<Tensor, Tensor?> _accumulatedGradients = new();
    private readonly HashSet<Tensor> _accumulatingTensors = new();
    private AccumulationContext? _context;
    private bool _disposed;

    /// <summary>
    /// Gets the number of accumulation steps configured.
    /// </summary>
    public int AccumulationCount { get; }

    /// <summary>
    /// Gets or sets whether gradient accumulation is enabled.
    /// </summary>
    public bool Enabled { get; set; }

    /// <summary>
    /// Gets whether the current accumulation cycle is complete and gradients are ready.
    /// </summary>
    public bool IsReady => _context != null && _context.IsReady;

    /// <summary>
    /// Gets the current number of accumulation steps completed.
    /// </summary>
    public int CurrentSteps => _context?.AccumulationSteps ?? 0;

    /// <summary>
    /// Initializes a new instance of the GradientAccumulator class.
    /// </summary>
    /// <param name="accumulationCount">The number of steps to accumulate gradients before applying.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when accumulationCount is less than 1.</exception>
    public GradientAccumulator(int accumulationCount)
    {
        if (accumulationCount < 1)
            throw new ArgumentOutOfRangeException(nameof(accumulationCount), "Accumulation count must be at least 1");

        AccumulationCount = accumulationCount;
        Enabled = true;
    }

    /// <summary>
    /// Enables gradient accumulation for the specified tensors.
    /// </summary>
    /// <param name="parameters">The tensors to accumulate gradients for.</param>
    /// <exception cref="ArgumentNullException">Thrown when parameters is null.</exception>
    public void EnableAccumulation(Tensor[] parameters)
    {
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        _accumulatingTensors.Clear();
        _accumulatedGradients.Clear();

        foreach (var tensor in parameters)
        {
            if (tensor == null)
                continue;

            if (!tensor.RequiresGrad)
                tensor.RequiresGrad = true;

            if (tensor.Gradient == null)
                tensor.Gradient = Tensor.Zeros(tensor.Shape);

            _accumulatingTensors.Add(tensor);
            _accumulatedGradients[tensor] = null;

            // Hook into the tensor's gradient accumulation
            RegisterGradientAccumulation(tensor);
        }

        _context = new AccumulationContext(AccumulationCount);
        Enabled = true;
    }

    /// <summary>
    /// Disables gradient accumulation and clears all state.
    /// </summary>
    public void DisableAccumulation()
    {
        foreach (var tensor in _accumulatingTensors)
        {
            if (tensor.Gradient != null)
            {
                tensor.Gradient = Tensor.Zeros(tensor.Shape);
            }
        }

        _accumulatingTensors.Clear();
        _accumulatedGradients.Clear();
        _context?.Dispose();
        _context = null;
        Enabled = false;
    }

    /// <summary>
    /// Increments the accumulation step counter.
    /// </summary>
    public void Step()
    {
        if (_context == null)
            throw new InvalidOperationException("Accumulation not enabled. Call EnableAccumulation first.");

        _context.Step();
    }

    /// <summary>
    /// Applies gradients using the specified optimizer action, scaled by the accumulation factor.
    /// </summary>
    /// <param name="optimizerStep">The action to apply gradients to each tensor.</param>
    /// <exception cref="ArgumentNullException">Thrown when optimizerStep is null.</exception>
    public void ApplyGradients(Action<Tensor> optimizerStep)
    {
        if (optimizerStep == null)
            throw new ArgumentNullException(nameof(optimizerStep));

        if (_context == null)
            throw new InvalidOperationException("Accumulation not enabled. Call EnableAccumulation first.");

        // Scale gradients by 1/accumulation_steps
        float scalingFactor = _context.ScalingFactor;

        foreach (var tensor in _accumulatingTensors)
        {
            if (tensor.Gradient != null)
            {
                // Scale the accumulated gradients
                for (int i = 0; i < tensor.Gradient.Data.Length; i++)
                {
                    tensor.Gradient.Data[i] *= scalingFactor;
                }

                // Apply optimizer step
                optimizerStep(tensor);
            }
        }

        // Reset for next accumulation cycle
        ResetGradients();
        _context.Reset();
    }

    /// <summary>
    /// Resets all accumulated gradients to zero.
    /// </summary>
    public void ResetGradients()
    {
        foreach (var tensor in _accumulatingTensors)
        {
            if (tensor.Gradient != null)
            {
                Array.Clear(tensor.Gradient.Data, 0, tensor.Gradient.Data.Length);
            }
        }
    }

    private void RegisterGradientAccumulation(Tensor tensor)
    {
        // The gradient accumulation happens naturally through the tensor's Backward method
        // which adds to existing gradients rather than replacing them.
        // We just need to ensure gradients are zeroed at the right times.
    }

    /// <summary>
    /// Disposes of the accumulator and releases resources.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            DisableAccumulation();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
