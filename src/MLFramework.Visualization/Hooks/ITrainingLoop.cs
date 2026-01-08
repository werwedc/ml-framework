using System;

namespace MLFramework.Visualization.Hooks;

/// <summary>
/// Interface for training loops that support hooks.
/// Hooks can be added to automatically log metrics, profile performance, etc.
/// </summary>
public interface ITrainingLoop
{
    /// <summary>
    /// Adds a hook to the training loop
    /// </summary>
    /// <param name="hook">The hook to add</param>
    void AddHook(ITrainingHook hook);

    /// <summary>
    /// Removes a hook from the training loop
    /// </summary>
    /// <param name="hook">The hook to remove</param>
    void RemoveHook(ITrainingHook hook);

    /// <summary>
    /// Trains the model for the specified number of epochs
    /// </summary>
    /// <param name="epochs">Number of epochs to train for</param>
    /// <param name="model">The model to train</param>
    /// <param name="dataLoader">The data loader for training data</param>
    void Train(int epochs, object model, object dataLoader);

    /// <summary>
    /// Validates the model using the provided data loader
    /// </summary>
    /// <param name="model">The model to validate</param>
    /// <param name="dataLoader">The data loader for validation data</param>
    void Validate(object model, object dataLoader);
}
