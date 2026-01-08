using System;

namespace MLFramework.HAL.CUDA;

/// <summary>
/// Represents a fixed memory block for graph execution.
/// </summary>
public class GraphMemoryBlock : IDisposable
{
    private bool _disposed;

    /// <summary>
    /// Gets the unique block identifier.
    /// </summary>
    public ulong BlockId { get; }

    /// <summary>
    /// Gets the pointer to the allocated memory.
    /// </summary>
    public IntPtr Ptr { get; }

    /// <summary>
    /// Gets the size of the block in bytes.
    /// </summary>
    public ulong Size { get; }

    /// <summary>
    /// Gets or sets whether the block is currently in use.
    /// This is used internally by the memory pool.
    /// </summary>
    internal bool InUse { get; set; }

    /// <summary>
    /// Initializes a new instance of the GraphMemoryBlock class.
    /// </summary>
    /// <param name="ptr">Pointer to the allocated memory</param>
    /// <param name="size">Size of the block in bytes</param>
    internal GraphMemoryBlock(IntPtr ptr, ulong size)
    {
        BlockId = GenerateBlockId();
        Ptr = ptr;
        Size = size;
        InUse = true;
        _disposed = false;
    }

    private static ulong _nextBlockId = 1;
    private static readonly object _blockIdLock = new object();

    private static ulong GenerateBlockId()
    {
        lock (_blockIdLock)
        {
            return _nextBlockId++;
        }
    }

    /// <summary>
    /// Disposes the block.
    /// Note: Memory is freed by the pool, not the block.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        // Note: Memory is freed by the pool, not the block
    }
}
