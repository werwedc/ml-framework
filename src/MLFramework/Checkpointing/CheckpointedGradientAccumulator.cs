using System;
using System.Collections.Generic;
using System.Linq;
using RitterTensor = RitterFramework.Core.Tensor;

namespace MLFramework.Checkpointing;

/// <summary>
/// Gradient accumulator that works with checkpointed activations
/// </summary>
public class CheckpointedGradientAccumulator : IDisposable
{
    private readonly int _accumulationSteps;
    private int _currentStep;
    private readonly Dictionary<string, RitterTensor.Tensor> _accumulatedGradients;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of CheckpointedGradientAccumulator
    /// </summary>
    /// <param name="accumulationSteps">Number of steps to accumulate</param>
    public CheckpointedGradientAccumulator(int accumulationSteps)
    {
        if (accumulationSteps <= 0)
            throw new ArgumentException("Accumulation steps must be greater than 0", nameof(accumulationSteps));

        _accumulationSteps = accumulationSteps;
        _currentStep = 0;
        _accumulatedGradients = new Dictionary<string, RitterTensor.Tensor>();
        _disposed = false;
    }

    /// <summary>
    /// Accumulates gradients for a batch
    /// </summary>
    /// <param name="gradients">Gradients to accumulate</param>
    public void Accumulate(Dictionary<string, RitterTensor.Tensor> gradients)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));
        ThrowIfDisposed();

        foreach (var kvp in gradients)
        {
            var paramName = kvp.Key;
            var gradient = kvp.Value;

            if (gradient == null)
                continue;

            if (!_accumulatedGradients.ContainsKey(paramName))
            {
                // Initialize gradient with zeros if not present
                _accumulatedGradients[paramName] = RitterTensor.Tensor.Zeros(gradient.Shape);
            }

            var accumulated = _accumulatedGradients[paramName];

            // Accumulate gradients: accumulated = accumulated + gradient
            for (int i = 0; i < accumulated.Data.Length; i++)
            {
                accumulated.Data[i] += gradient.Data[i];
            }
        }

        _currentStep++;
    }

    /// <summary>
    /// Gets the accumulated gradients and resets the accumulator
    /// </summary>
    /// <returns>Accumulated gradients</returns>
    public Dictionary<string, RitterTensor.Tensor> GetAccumulatedGradients()
    {
        ThrowIfDisposed();

        if (_currentStep == 0)
        {
            return new Dictionary<string, RitterTensor.Tensor>();
        }

        // Divide accumulated gradients by the number of accumulation steps
        var result = new Dictionary<string, RitterTensor.Tensor>();
        foreach (var kvp in _accumulatedGradients)
        {
            var paramName = kvp.Key;
            var gradient = kvp.Value;

            // Create a new tensor with averaged gradients
            var averagedData = new float[gradient.Data.Length];
            for (int i = 0; i < gradient.Data.Length; i++)
            {
                averagedData[i] = gradient.Data[i] / _currentStep;
            }

            var averagedGradient = new RitterTensor.Tensor(averagedData, gradient.Shape, false);
            result[paramName] = averagedGradient;
        }

        Reset();
        return result;
    }

    /// <summary>
    /// Resets the accumulator without returning gradients
    /// </summary>
    public void Reset()
    {
        ThrowIfDisposed();

        lock (_accumulatedGradients)
        {
            _accumulatedGradients.Clear();
            _currentStep = 0;
        }
    }

    /// <summary>
    /// Checks if it's time to apply accumulated gradients
    /// </summary>
    /// <returns>True if accumulation steps reached, false otherwise</returns>
    public bool ShouldApplyGradients()
    {
        ThrowIfDisposed();
        return _currentStep >= _accumulationSteps;
    }

    /// <summary>
    /// Gets the current accumulation step
    /// </summary>
    public int CurrentStep => _currentStep;

    /// <summary>
    /// Gets the number of accumulation steps
    /// </summary>
    public int AccumulationSteps => _accumulationSteps;

    /// <summary>
    /// Gets whether there are any accumulated gradients
    /// </summary>
    public bool HasGradients => _accumulatedGradients.Count > 0;

    /// <summary>
    /// Gets the number of parameters being accumulated
    /// </summary>
    public int ParameterCount
    {
        get
        {
            ThrowIfDisposed();
            return _accumulatedGradients.Count;
        }
    }

    /// <summary>
    /// Gets a copy of the accumulated gradients without resetting
    /// </summary>
    /// <returns>Copy of accumulated gradients</returns>
    public Dictionary<string, RitterTensor.Tensor> PeekGradients()
    {
        ThrowIfDisposed();

        if (_currentStep == 0)
        {
            return new Dictionary<string, RitterTensor.Tensor>();
        }

        var result = new Dictionary<string, RitterTensor.Tensor>();
        foreach (var kvp in _accumulatedGradients)
        {
            var paramName = kvp.Key;
            var gradient = kvp.Value;

            // Clone the gradient tensor
            var clonedData = new float[gradient.Data.Length];
            Array.Copy(gradient.Data, clonedData, gradient.Data.Length);

            var clonedGradient = new RitterTensor.Tensor(clonedData, gradient.Shape, false);
            result[paramName] = clonedGradient;
        }

        return result;
    }

    /// <summary>
    /// Removes gradients for a specific parameter
    /// </summary>
    /// <param name="paramName">Name of the parameter</param>
    public void RemoveGradient(string paramName)
    {
        if (string.IsNullOrEmpty(paramName))
            throw new ArgumentException("Parameter name cannot be null or empty", nameof(paramName));
        ThrowIfDisposed();

        lock (_accumulatedGradients)
        {
            if (_accumulatedGradients.TryGetValue(paramName, out var gradient))
            {
                _accumulatedGradients.Remove(paramName);
            }
        }
    }

    /// <summary>
    /// Gets the gradient for a specific parameter
    /// </summary>
    /// <param name="paramName">Name of the parameter</param>
    /// <returns>Gradient tensor or null if not found</returns>
    public RitterTensor.Tensor? GetGradient(string paramName)
    {
        if (string.IsNullOrEmpty(paramName))
            throw new ArgumentException("Parameter name cannot be null or empty", nameof(paramName));
        ThrowIfDisposed();

        lock (_accumulatedGradients)
        {
            if (_accumulatedGradients.TryGetValue(paramName, out var gradient))
            {
                // Clone the gradient tensor
                var clonedData = new float[gradient.Data.Length];
                Array.Copy(gradient.Data, clonedData, gradient.Data.Length);

                return new RitterTensor.Tensor(clonedData, gradient.Shape, false);
            }
        }

        return null;
    }

    /// <summary>
    /// Disposes the accumulator and releases resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            Reset();
            _disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CheckpointedGradientAccumulator));
        }
    }
}
