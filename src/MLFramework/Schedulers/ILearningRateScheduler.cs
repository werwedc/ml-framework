namespace MLFramework.Schedulers;

/// <summary>
/// Defines the contract for learning rate schedulers.
/// </summary>
public interface ILearningRateScheduler
{
    /// <summary>
    /// Gets the learning rate for the current step.
    /// </summary>
    /// <param name="step">Current training step/iteration.</param>
    /// <param name="baseLearningRate">Base learning rate provided by the optimizer.</param>
    /// <returns>Learning rate to use for this step.</returns>
    float GetLearningRate(int step, float baseLearningRate);

    /// <summary>
    /// Advances the scheduler state by one step.
    /// Called after each optimizer step.
    /// </summary>
    void Step();

    /// <summary>
    /// Resets the scheduler to its initial state.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets the current state of the scheduler for checkpointing.
    /// </summary>
    StateDict GetState();

    /// <summary>
    /// Loads the scheduler state from a checkpoint.
    /// </summary>
    /// <param name="state">State dictionary to load from.</param>
    void LoadState(StateDict state);
}
