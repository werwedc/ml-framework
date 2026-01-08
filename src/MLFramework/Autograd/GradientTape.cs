using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Represents the gradient retention policy for higher-order derivative computation.
/// </summary>
public enum GradientRetentionPolicy
{
    /// <summary>
    /// Discard gradients after backward pass (default behavior for first-order differentiation).
    /// </summary>
    Discard,

    /// <summary>
    /// Keep all gradients to enable higher-order differentiation.
    /// </summary>
    Keep,

    /// <summary>
    /// Selectively retain only gradients that will be differentiated.
    /// </summary>
    Selective
}

/// <summary>
/// Context for recording computation graphs with support for higher-order derivatives.
/// Enables nested differentiation contexts for computing gradients of gradients.
/// </summary>
public class GradientTape : IDisposable
{
    private bool _disposed;
    private bool _higherOrderEnabled;
    private GradientRetentionPolicy _retentionPolicy;
    private List<Tensor> _watchedTensors;
    private Dictionary<Tensor, Tensor> _gradientHistory;
    private GradientTape? _parentTape;
    private static readonly object _lock = new object();
    private static int _globalTapeDepth;

    /// <summary>
    /// Gets or sets whether higher-order tracking is enabled.
    /// When true, gradients themselves can be differentiated.
    /// </summary>
    public bool HigherOrderEnabled => _higherOrderEnabled;

    /// <summary>
    /// Gets the current gradient retention policy.
    /// </summary>
    public GradientRetentionPolicy RetentionPolicy => _retentionPolicy;

    /// <summary>
    /// Gets the nesting depth of this tape (0 = root tape).
    /// </summary>
    public int Depth { get; private set; }

    /// <summary>
    /// Gets the number of tensors currently being watched by this tape.
    /// </summary>
    public int WatchedTensorCount => _watchedTensors.Count;

    /// <summary>
    /// Private constructor for creating tape instances.
    /// </summary>
    /// <param name="higherOrderEnabled">Whether to enable higher-order tracking.</param>
    /// <param name="retentionPolicy">The gradient retention policy to use.</param>
    /// <param name="parentTape">The parent tape for nested contexts.</param>
    private GradientTape(bool higherOrderEnabled, GradientRetentionPolicy retentionPolicy, GradientTape? parentTape)
    {
        _higherOrderEnabled = higherOrderEnabled;
        _retentionPolicy = retentionPolicy;
        _parentTape = parentTape;
        _watchedTensors = new List<Tensor>();
        _gradientHistory = new Dictionary<Tensor, Tensor>();
        Depth = parentTape != null ? parentTape.Depth + 1 : 0;

        lock (_lock)
        {
            _globalTapeDepth = Depth;
        }
    }

    /// <summary>
    /// Creates a new gradient tape with higher-order tracking enabled.
    /// This allows gradients to be differentiated themselves for computing higher-order derivatives.
    /// </summary>
    /// <returns>A new GradientTape instance with higher-order tracking enabled.</returns>
    public static GradientTape EnableHigherOrderTracking()
    {
        return new GradientTape(
            higherOrderEnabled: true,
            retentionPolicy: GradientRetentionPolicy.Keep,
            parentTape: null
        );
    }

    /// <summary>
    /// Creates a new gradient tape with specified settings.
    /// </summary>
    /// <param name="higherOrderEnabled">Whether to enable higher-order tracking (default: false).</param>
    /// <param name="retentionPolicy">The gradient retention policy (default: Discard).</param>
    /// <returns>A new GradientTape instance.</returns>
    public static GradientTape Create(bool higherOrderEnabled = false, GradientRetentionPolicy retentionPolicy = GradientRetentionPolicy.Discard)
    {
        return new GradientTape(
            higherOrderEnabled: higherOrderEnabled,
            retentionPolicy: retentionPolicy,
            parentTape: null
        );
    }

    /// <summary>
    /// Creates a nested tape context for gradient-of-gradient computation.
    /// The nested tape inherits settings from the parent tape.
    /// </summary>
    /// <returns>A new GradientTape instance nested within the current tape.</returns>
    /// <exception cref="InvalidOperationException">Thrown when higher-order tracking is not enabled.</exception>
    public GradientTape Record()
    {
        if (!_higherOrderEnabled)
        {
            throw new InvalidOperationException("Higher-order tracking must be enabled to create nested tapes. Call EnableHigherOrderTracking() first.");
        }

        return new GradientTape(
            higherOrderEnabled: true,
            retentionPolicy: _retentionPolicy,
            parentTape: this
        );
    }

    /// <summary>
    /// Watches a tensor to track its computation graph for gradient computation.
    /// </summary>
    /// <param name="tensor">The tensor to watch.</param>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public void Watch(Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!_watchedTensors.Contains(tensor))
        {
            _watchedTensors.Add(tensor);

            // Enable gradient tracking if not already enabled
            if (!tensor.RequiresGrad)
            {
                // Note: This would require extending the Tensor class to support
                // dynamically changing RequiresGrad. For now, we'll throw.
                throw new ArgumentException("Tensor must have RequiresGrad=true to be watched. Enable gradients before watching.");
            }
        }
    }

    /// <summary>
    /// Computes gradients of the loss with respect to the watched tensors.
    /// </summary>
    /// <param name="loss">The loss tensor (must be scalar).</param>
    /// <param name="gradientOutput">Optional gradient to backpropagate (for non-scalar losses).</param>
    /// <returns>A dictionary mapping watched tensors to their gradients.</returns>
    /// <exception cref="ArgumentNullException">Thrown when loss is null.</exception>
    /// <exception cref="ArgumentException">Thrown when loss is not scalar and gradientOutput is not provided.</exception>
    public Dictionary<Tensor, Tensor> Gradient(Tensor loss, Tensor? gradientOutput = null)
    {
        if (loss == null)
            throw new ArgumentNullException(nameof(loss));

        if (loss.Size != 1 && gradientOutput == null)
            throw new ArgumentException("Loss must be scalar (size 1) unless gradientOutput is provided");

        // Compute gradients using backward pass
        loss.Backward(gradientOutput);

        var gradients = new Dictionary<Tensor, Tensor>();

        // Collect gradients from watched tensors
        foreach (var tensor in _watchedTensors)
        {
            if (tensor.Gradient != null)
            {
                if (_higherOrderEnabled && _retentionPolicy == GradientRetentionPolicy.Keep)
                {
                    // Store gradient for potential higher-order differentiation
                    // Clone the gradient and enable gradient tracking
                    var gradCopy = CloneWithGrad(tensor.Gradient);
                    _gradientHistory[tensor] = gradCopy;
                    gradients[tensor] = gradCopy;
                }
                else
                {
                    gradients[tensor] = tensor.Gradient;
                }
            }
        }

        return gradients;
    }

    /// <summary>
    /// Gets the gradient history for a specific tensor.
    /// Used for higher-order derivative computation.
    /// </summary>
    /// <param name="tensor">The tensor to get gradient history for.</param>
    /// <returns>The gradient tensor, or null if not found.</returns>
    public Tensor? GetGradientHistory(Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        _gradientHistory.TryGetValue(tensor, out var gradient);
        return gradient;
    }

    /// <summary>
    /// Clears all watched tensors and gradient history.
    /// </summary>
    public void Reset()
    {
        _watchedTensors.Clear();
        _gradientHistory.Clear();
    }

    /// <summary>
    /// Enables gradient checkpointing for memory efficiency in large models.
    /// This trades computation for memory by recomputing intermediate values during backward pass.
    /// </summary>
    public void EnableCheckpointing()
    {
        // In a full implementation, this would configure the tape to use
        // gradient checkpointing strategies. For now, it's a placeholder.
        // Implementation would involve:
        // 1. Storing only checkpoints (subset of intermediate activations)
        // 2. Recomputing missing activations during backward pass
        // 3. Managing checkpoint selection algorithms
    }

    /// <summary>
    /// Merges the computation graphs from nested tapes into the parent tape.
    /// This is used when exiting nested contexts to preserve gradient history.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when there is no parent tape.</exception>
    public void MergeToParent()
    {
        if (_parentTape == null)
            throw new InvalidOperationException("Cannot merge to parent: no parent tape exists");

        // Merge gradient history to parent tape
        foreach (var kvp in _gradientHistory)
        {
            if (!_parentTape._gradientHistory.ContainsKey(kvp.Key))
            {
                _parentTape._gradientHistory[kvp.Key] = kvp.Value;
            }
        }

        // Merge watched tensors
        foreach (var tensor in _watchedTensors)
        {
            if (!_parentTape._watchedTensors.Contains(tensor))
            {
                _parentTape._watchedTensors.Add(tensor);
            }
        }
    }

    /// <summary>
    /// Gets the current global tape depth (number of nested tapes).
    /// </summary>
    /// <returns>The current tape depth.</returns>
    public static int GetGlobalDepth()
    {
        lock (_lock)
        {
            return _globalTapeDepth;
        }
    }

    /// <summary>
    /// Disposes the GradientTape and releases resources.
    /// If this is a nested tape, it merges to parent before disposal.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Protected dispose implementation.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                // Merge to parent if this is a nested tape
                if (_parentTape != null)
                {
                    MergeToParent();
                }

                // Clear resources
                _watchedTensors.Clear();
                _gradientHistory.Clear();

                // Update global depth
                lock (_lock)
                {
                    _globalTapeDepth = Depth > 0 ? Depth - 1 : 0;
                }
            }

            _disposed = true;
        }
    }

    /// <summary>
    /// Finalizer.
    /// </summary>
    ~GradientTape()
    {
        Dispose(false);
    }

    /// <summary>
    /// Clones a tensor and enables gradient tracking.
    /// </summary>
    private static Tensor CloneWithGrad(Tensor tensor)
    {
        // Use TensorAccessor to clone with gradient tracking
        return TensorAccessor.CloneWithGrad(tensor);
    }
}
