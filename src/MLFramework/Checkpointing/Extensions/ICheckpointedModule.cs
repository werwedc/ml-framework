namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Interface for checkpointed modules
/// </summary>
/// <typeparam name="T">Type of the underlying module</typeparam>
public interface ICheckpointedModule<T> : IDisposable
{
    /// <summary>
    /// Gets the underlying module
    /// </summary>
    T Module { get; }

    /// <summary>
    /// Gets the layer ID
    /// </summary>
    string LayerId { get; }

    /// <summary>
    /// Gets the checkpoint configuration
    /// </summary>
    CheckpointConfig Config { get; }

    /// <summary>
    /// Enables checkpointing
    /// </summary>
    void EnableCheckpointing();

    /// <summary>
    /// Disables checkpointing
    /// </summary>
    void DisableCheckpointing();

    /// <summary>
    /// Gets checkpointing statistics
    /// </summary>
    /// <returns>Checkpointing statistics</returns>
    CheckpointStatistics GetStatistics();
}
