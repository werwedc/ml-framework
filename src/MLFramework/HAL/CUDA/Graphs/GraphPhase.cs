namespace MLFramework.HAL.CUDA.Graphs;

/// <summary>
/// Defines common training phases for CUDA graphs.
/// </summary>
public enum GraphPhase
{
    /// <summary>
    /// Forward pass only
    /// </summary>
    Forward,

    /// <summary>
    /// Backward pass only
    /// </summary>
    Backward,

    /// <summary>
    /// Optimizer step only
    /// </summary>
    OptimizerStep,

    /// <summary>
    /// Forward and backward passes combined
    /// </summary>
    ForwardBackward,

    /// <summary>
    /// Complete training step including forward, backward, and optimizer
    /// </summary>
    FullTrainingStep
}
