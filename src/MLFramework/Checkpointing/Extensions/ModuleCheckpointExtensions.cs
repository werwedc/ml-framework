using System;

namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Extension methods for checkpointing individual modules/layers
/// </summary>
public static class ModuleCheckpointExtensions
{
    /// <summary>
    /// Wraps a module with checkpointing
    /// </summary>
    /// <typeparam name="T">Type of the module</typeparam>
    /// <param name="module">The module to wrap</param>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>Checkpointed module wrapper</returns>
    public static ICheckpointedModule<T> AsCheckpointed<T>(this T module, string layerId)
        where T : class
    {
        return new CheckpointedModuleWrapper<T>(module, layerId);
    }

    /// <summary>
    /// Wraps a module with checkpointing and custom configuration
    /// </summary>
    /// <typeparam name="T">Type of the module</typeparam>
    /// <param name="module">The module to wrap</param>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="config">Checkpoint configuration</param>
    /// <returns>Checkpointed module wrapper</returns>
    public static ICheckpointedModule<T> AsCheckpointed<T>(
        this T module,
        string layerId,
        CheckpointConfig config)
        where T : class
    {
        return new CheckpointedModuleWrapper<T>(module, layerId, config);
    }
}
