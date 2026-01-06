using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Context for managing gradient accumulation state across multiple forward/backward passes.
/// Enables efficient training with large effective batch sizes that don't fit in memory.
/// </summary>
public class AccumulationContext : IDisposable
{
    private readonly HashSet<Tensor> _registeredTensors = new();
    private bool _disposed;

    /// <summary>
    /// Gets the current number of accumulation steps completed.
    /// </summary>
    public int AccumulationSteps { get; private set; }

    /// <summary>
    /// Gets the target number of steps before gradients are ready to be applied.
    /// </summary>
    public int TargetSteps { get; }

    /// <summary>
    /// Gets whether the accumulation cycle is complete and gradients are ready.
    /// </summary>
    public bool IsReady => AccumulationSteps >= TargetSteps;

    /// <summary>
    /// Gets the scaling factor for accumulated gradients (1/target_steps).
    /// </summary>
    public float ScalingFactor => 1.0f / TargetSteps;

    /// <summary>
    /// Initializes a new instance of the AccumulationContext class.
    /// </summary>
    /// <param name="targetSteps">The number of accumulation steps before applying gradients.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when targetSteps is less than 1.</exception>
    public AccumulationContext(int targetSteps)
    {
        if (targetSteps < 1)
            throw new ArgumentOutOfRangeException(nameof(targetSteps), "Target steps must be at least 1");

        TargetSteps = targetSteps;
        AccumulationSteps = 0;
    }

    /// <summary>
    /// Registers a tensor for gradient accumulation.
    /// </summary>
    /// <param name="tensor">The tensor to register.</param>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public void RegisterTensor(Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        _registeredTensors.Add(tensor);
    }

    /// <summary>
    /// Increments the accumulation step counter.
    /// </summary>
    /// <exception cref="ObjectDisposedException">Thrown when the context has been disposed.</exception>
    public void Step()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AccumulationContext));

        AccumulationSteps++;
    }

    /// <summary>
    /// Resets the accumulation counter to start a new accumulation cycle.
    /// </summary>
    /// <exception cref="ObjectDisposedException">Thrown when the context has been disposed.</exception>
    public void Reset()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AccumulationContext));

        AccumulationSteps = 0;
    }

    /// <summary>
    /// Disposes of the context and clears registered tensors.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _registeredTensors.Clear();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
