namespace MLFramework.Visualization.Hooks;

/// <summary>
/// Represents different phases in the training loop.
/// Hooks can be triggered at the start or end of each phase.
/// </summary>
public enum TrainingPhase
{
    /// <summary>
    /// Start of an epoch
    /// </summary>
    EpochStart,

    /// <summary>
    /// End of an epoch
    /// </summary>
    EpochEnd,

    /// <summary>
    /// Start of a batch iteration
    /// </summary>
    BatchStart,

    /// <summary>
    /// End of a batch iteration
    /// </summary>
    BatchEnd,

    /// <summary>
    /// Start of forward pass
    /// </summary>
    ForwardPassStart,

    /// <summary>
    /// End of forward pass
    /// </summary>
    ForwardPassEnd,

    /// <summary>
    /// Start of backward pass (gradient computation)
    /// </summary>
    BackwardPassStart,

    /// <summary>
    /// End of backward pass
    /// </summary>
    BackwardPassEnd,

    /// <summary>
    /// Optimizer step (parameter update)
    /// </summary>
    OptimizerStep,

    /// <summary>
    /// Start of validation phase
    /// </summary>
    ValidationStart,

    /// <summary>
    /// End of validation phase
    /// </summary>
    ValidationEnd,

    /// <summary>
    /// Checkpoint save operation
    /// </summary>
    CheckpointSave
}
