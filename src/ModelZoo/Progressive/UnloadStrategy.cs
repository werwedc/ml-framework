namespace MLFramework.ModelZoo.Progressive;

/// <summary>
/// Defines when layers should be unloaded from memory.
/// </summary>
public enum UnloadStrategy
{
    /// <summary>
    /// Never unload layers once loaded.
    /// </summary>
    Never,

    /// <summary>
    /// Unload layers when memory pressure is detected.
    /// </summary>
    MemoryPressure,

    /// <summary>
    /// Use Least Recently Used (LRU) policy to unload old layers.
    /// </summary>
    LRU
}
