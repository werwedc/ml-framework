using System;

namespace MLFramework.HAL.CUDA;

/// <summary>
/// Extension methods for memory manager operations.
/// </summary>
public static class MemoryManagerExtensions
{
    /// <summary>
    /// Configures the allocator for graph capture.
    /// </summary>
    /// <param name="allocator">The allocator to configure</param>
    /// <param name="pool">The graph memory pool to use</param>
    public static void ConfigureForGraph(
        this ICUDAMemoryAllocator allocator,
        CUDAGraphMemoryPool pool)
    {
        allocator.GraphPool = pool;
        allocator.SetGraphMode(true);
    }

    /// <summary>
    /// Enables graph capture mode.
    /// </summary>
    /// <param name="allocator">The allocator to configure</param>
    public static void EnableGraphMode(this ICUDAMemoryAllocator allocator)
    {
        allocator.SetGraphMode(true);
    }

    /// <summary>
    /// Disables graph capture mode.
    /// </summary>
    /// <param name="allocator">The allocator to configure</param>
    public static void DisableGraphMode(this ICUDAMemoryAllocator allocator)
    {
        allocator.SetGraphMode(false);
    }

    /// <summary>
    /// Executes an action with graph mode enabled.
    /// </summary>
    /// <typeparam name="T">The return type</typeparam>
    /// <param name="allocator">The allocator to use</param>
    /// <param name="action">The action to execute</param>
    /// <returns>The result of the action</returns>
    public static T WithGraphMode<T>(
        this ICUDAMemoryAllocator allocator,
        Func<T> action)
    {
        var wasGraphMode = allocator.IsGraphMode;
        try
        {
            allocator.EnableGraphMode();
            return action();
        }
        finally
        {
            if (!wasGraphMode)
            {
                allocator.DisableGraphMode();
            }
        }
    }

    /// <summary>
    /// Executes an action with graph mode enabled.
    /// </summary>
    /// <param name="allocator">The allocator to use</param>
    /// <param name="action">The action to execute</param>
    public static void WithGraphMode(
        this ICUDAMemoryAllocator allocator,
        Action action)
    {
        var wasGraphMode = allocator.IsGraphMode;
        try
        {
            allocator.EnableGraphMode();
            action();
        }
        finally
        {
            if (!wasGraphMode)
            {
                allocator.DisableGraphMode();
            }
        }
    }
}
