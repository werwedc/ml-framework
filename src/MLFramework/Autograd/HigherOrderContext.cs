using System;

namespace MLFramework.Autograd;

/// <summary>
/// Context for managing computation graph retention during higher-order derivative computation.
/// This context ensures that intermediate computation graphs are preserved for subsequent backward passes.
/// </summary>
public class HigherOrderContext : IDisposable
{
    private bool _disposed;
    private static readonly object _lock = new object();

    /// <summary>
    /// Gets or sets whether to create and retain computation graphs for higher-order derivatives.
    /// When true, graphs are retained after backward passes to enable gradient-of-gradient computation.
    /// </summary>
    public bool CreateGraph { get; set; }

    /// <summary>
    /// Gets or sets the maximum order of derivatives to support.
    /// For example, 2 supports first and second-order derivatives (gradients and Hessians).
    /// </summary>
    public int MaxOrder { get; set; }

    /// <summary>
    /// Gets the current derivative order being computed.
    /// </summary>
    public int CurrentOrder { get; private set; }

    /// <summary>
    /// Initializes a new instance of the HigherOrderContext class.
    /// </summary>
    /// <param name="createGraph">Whether to create and retain computation graphs (default: true).</param>
    /// <param name="maxOrder">Maximum order of derivatives to support (default: 2).</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when maxOrder is less than 1.</exception>
    public HigherOrderContext(bool createGraph = true, int maxOrder = 2)
    {
        if (maxOrder < 1)
            throw new ArgumentOutOfRangeException(nameof(maxOrder), "MaxOrder must be at least 1");

        CreateGraph = createGraph;
        MaxOrder = maxOrder;
        CurrentOrder = 0;

        if (createGraph)
        {
            EnableGraphRetention();
        }
    }

    /// <summary>
    /// Enables graph retention to preserve computation graphs for higher-order derivatives.
    /// </summary>
    public void EnableGraphRetention()
    {
        CreateGraph = true;
        // In a full implementation, this would set global flags or modify tensor behavior
        // For now, it's a property that consumers check before calling backward
    }

    /// <summary>
    /// Disables graph retention, allowing the computation graph to be freed after backward pass.
    /// </summary>
    public void DisableGraphRetention()
    {
        CreateGraph = false;
    }

    /// <summary>
    /// Increments the current derivative order.
    /// Called before computing higher-order derivatives.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when attempting to exceed MaxOrder.</exception>
    public void IncrementOrder()
    {
        if (CurrentOrder >= MaxOrder)
            throw new InvalidOperationException($"Cannot compute derivative order {CurrentOrder + 1}. MaxOrder is set to {MaxOrder}");

        lock (_lock)
        {
            CurrentOrder++;
        }
    }

    /// <summary>
    /// Resets the current derivative order to zero.
    /// </summary>
    public void ResetOrder()
    {
        lock (_lock)
        {
            CurrentOrder = 0;
        }
    }

    /// <summary>
    /// Determines if graphs should be retained based on current order and max order.
    /// </summary>
    /// <returns>True if graphs should be retained, false otherwise.</returns>
    public bool ShouldRetainGraph()
    {
        return CreateGraph && (CurrentOrder < MaxOrder);
    }

    /// <summary>
    /// Disposes the HigherOrderContext and releases any resources.
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
                // Release any managed resources
                DisableGraphRetention();
                ResetOrder();
            }

            _disposed = true;
        }
    }

    /// <summary>
    /// Finalizer.
    /// </summary>
    ~HigherOrderContext()
    {
        Dispose(false);
    }
}
