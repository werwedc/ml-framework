namespace MLFramework.Compilation;

/// <summary>
/// Represents a cache entry for a compiled kernel
/// </summary>
public class KernelCacheEntry
{
    /// <summary>
    /// Gets the shape signature for this entry
    /// </summary>
    public required ShapeSignature Signature { get; init; }

    /// <summary>
    /// Gets the compiled kernel
    /// </summary>
    public required object CompiledKernel { get; init; }

    /// <summary>
    /// Gets the last time this entry was accessed
    /// </summary>
    public DateTime LastUsed { get; private set; }

    /// <summary>
    /// Gets the number of times this entry has been used
    /// </summary>
    public int UseCount { get; private set; }

    /// <summary>
    /// Gets the time taken to compile this kernel in milliseconds
    /// </summary>
    public required long CompilationTimeMs { get; init; }

    /// <summary>
    /// Creates a new kernel cache entry
    /// </summary>
    public KernelCacheEntry()
    {
        LastUsed = DateTime.UtcNow;
        UseCount = 0;
    }

    /// <summary>
    /// Updates the last accessed time
    /// </summary>
    public void UpdateAccessTime()
    {
        LastUsed = DateTime.UtcNow;
    }

    /// <summary>
    /// Increments the use count
    /// </summary>
    public void IncrementUseCount()
    {
        UseCount++;
    }

    /// <summary>
    /// Creates a new cache entry with the specified values
    /// </summary>
    public static KernelCacheEntry Create(ShapeSignature signature, object compiledKernel, long compilationTimeMs)
    {
        return new KernelCacheEntry
        {
            Signature = signature,
            CompiledKernel = compiledKernel,
            CompilationTimeMs = compilationTimeMs
        };
    }
}
