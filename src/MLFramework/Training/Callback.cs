using System;
using System.Collections.Generic;

namespace MLFramework.Training;

/// <summary>
/// Base class for training callbacks. Callbacks can be used to inject custom logic
/// into the training loop at specific points (batch end, epoch end, validation end).
/// </summary>
public abstract class Callback
{
    /// <summary>
    /// Called at the end of each batch.
    /// </summary>
    /// <param name="batch">The current batch index (0-based).</param>
    /// <param name="metrics">Dictionary of metrics collected during this batch.</param>
    public virtual void OnBatchEnd(int batch, Dictionary<string, float> metrics)
    {
        // Default implementation: do nothing
    }

    /// <summary>
    /// Called at the end of each epoch.
    /// </summary>
    /// <param name="epoch">The current epoch index (0-based).</param>
    /// <param name="metrics">Dictionary of metrics collected during this epoch.</param>
    public virtual void OnEpochEnd(int epoch, Dictionary<string, float> metrics)
    {
        // Default implementation: do nothing
    }

    /// <summary>
    /// Called at the end of validation.
    /// </summary>
    /// <param name="metrics">Dictionary of validation metrics.</param>
    public virtual void OnValidationEnd(Dictionary<string, float> metrics)
    {
        // Default implementation: do nothing
    }

    /// <summary>
    /// Called at the beginning of training.
    /// </summary>
    /// <param name="model">The model being trained.</param>
    public virtual void OnTrainBegin(object model)
    {
        // Default implementation: do nothing
    }

    /// <summary>
    /// Called at the end of training.
    /// </summary>
    /// <param name="metrics">Final metrics dictionary.</param>
    public virtual void OnTrainEnd(Dictionary<string, float> metrics)
    {
        // Default implementation: do nothing
    }

    /// <summary>
    /// Called at the beginning of each epoch.
    /// </summary>
    /// <param name="epoch">The current epoch index (0-based).</param>
    public virtual void OnEpochBegin(int epoch)
    {
        // Default implementation: do nothing
    }

    /// <summary>
    /// Called at the beginning of each batch.
    /// </summary>
    /// <param name="batch">The current batch index (0-based).</param>
    public virtual void OnBatchBegin(int batch)
    {
        // Default implementation: do nothing
    }
}
