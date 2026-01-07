namespace MLFramework.Distributed.Communication;

using System;
using System.Collections.Generic;

/// <summary>
/// Factory for creating communication backends.
/// </summary>
public static class CommunicatorFactory
{
    /// <summary>
    /// Creates a communicator instance with the specified backend.
    /// </summary>
    /// <param name="backend">Backend type ("mock", "nccl", etc.)</param>
    /// <param name="config">Optional configuration dictionary</param>
    /// <returns>Configured communicator instance</returns>
    public static ICommunicator Create(string backend = "mock", Dictionary<string, object>? config = null)
    {
        if (string.IsNullOrEmpty(backend))
        {
            throw new ArgumentException("Backend name cannot be null or empty", nameof(backend));
        }

        return backend.ToLowerInvariant() switch
        {
            "mock" => CreateMockCommunicator(config),
            "nccl" => CreateNCCLCommunicator(config),
            _ => throw new ArgumentException($"Unknown backend: {backend}", nameof(backend))
        };
    }

    /// <summary>
    /// Creates a mock communicator for testing.
    /// </summary>
    /// <param name="config">Configuration dictionary with optional keys:
    /// - "world_size": Total number of processes (default: 1)
    /// - "rank": Rank of this process (default: 0)
    /// </param>
    private static ICommunicator CreateMockCommunicator(Dictionary<string, object>? config)
    {
        int worldSize = config?.GetValueOrDefault("world_size", 1) as int? ?? 1;
        int rank = config?.GetValueOrDefault("rank", 0) as int? ?? 0;

        return new MockCommunicator(worldSize, rank);
    }

    /// <summary>
    /// Creates an NCCL communicator for GPU-based distributed training.
    /// </summary>
    /// <param name="config">Configuration dictionary with optional keys:
    /// - "world_size": Total number of processes (default: 1)
    /// - "rank": Rank of this process (default: 0)
    /// - "device": GPU device ID (default: 0)
    /// </param>
    private static ICommunicator CreateNCCLCommunicator(Dictionary<string, object>? config)
    {
        int worldSize = config?.GetValueOrDefault("world_size", 1) as int? ?? 1;
        int rank = config?.GetValueOrDefault("rank", 0) as int? ?? 0;
        int device = config?.GetValueOrDefault("device", 0) as int? ?? 0;

        return new NCCLCommunicator(worldSize, rank, device);
    }
}
