using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;
using MLFramework.Modules;

namespace MLFramework.Checkpoint;

/// <summary>
/// Static utility class for managing tensor-parallel model checkpoints
/// </summary>
public static class TPCheckpointManager
{
    /// <summary>
    /// Save checkpoint (distributed format) - IModule version
    /// </summary>
    public static async Task SaveDistributedAsync(
        IModule model,
        string checkpointDir,
        string? checkpointName = null)
    {
        var rank = 0; // Would get from TensorParallel context
        var checkpoint = new DistributedTPCheckpoint(checkpointDir);
        await checkpoint.SaveAsync(model, rank, checkpointName);

        // Wait for all ranks to finish saving
        // await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Save checkpoint (distributed format) - Module version
    /// </summary>
    public static async Task SaveDistributedAsync(
        NN.Module model,
        string checkpointDir,
        string? checkpointName = null)
    {
        var rank = 0; // Would get from TensorParallel context
        var checkpoint = new DistributedTPCheckpoint(checkpointDir);
        await checkpoint.SaveAsync(model, rank, checkpointName);

        // Wait for all ranks to finish saving
        // await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Load checkpoint (distributed format) - IModule version
    /// </summary>
    public static async Task LoadDistributedAsync(
        IModule model,
        string checkpointDir)
    {
        var rank = 0; // Would get from TensorParallel context
        var checkpoint = new DistributedTPCheckpoint(checkpointDir);

        await checkpoint.LoadAsync(model, rank);

        // Wait for all ranks to finish loading
        // await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Load checkpoint (distributed format) - Module version
    /// </summary>
    public static async Task LoadDistributedAsync(
        NN.Module model,
        string checkpointDir)
    {
        var rank = 0; // Would get from TensorParallel context
        var checkpoint = new DistributedTPCheckpoint(checkpointDir);

        await checkpoint.LoadAsync(model, rank);

        // Wait for all ranks to finish loading
        // await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Save checkpoint (centralized format) - IModule version
    /// </summary>
    public static async Task SaveCentralizedAsync(
        IModule model,
        string checkpointFile,
        string? checkpointName = null)
    {
        var rank = 0; // Would get from TensorParallel context
        var worldSize = 1; // Would get from TensorParallel context

        var checkpoint = new CentralizedTPCheckpoint(checkpointFile);
        await checkpoint.SaveAsync(model, rank, worldSize, checkpointName);

        // Wait for all ranks to finish saving
        // await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Save checkpoint (centralized format) - Module version
    /// </summary>
    public static async Task SaveCentralizedAsync(
        NN.Module model,
        string checkpointFile,
        string? checkpointName = null)
    {
        var rank = 0; // Would get from TensorParallel context
        var worldSize = 1; // Would get from TensorParallel context

        var checkpoint = new CentralizedTPCheckpoint(checkpointFile);
        await checkpoint.SaveAsync(model, rank, worldSize, checkpointName);

        // Wait for all ranks to finish saving
        // await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Load checkpoint (centralized format) - IModule version
    /// </summary>
    public static async Task LoadCentralizedAsync(
        IModule model,
        string checkpointFile)
    {
        var rank = 0; // Would get from TensorParallel context
        var worldSize = 1; // Would get from TensorParallel context

        var checkpoint = new CentralizedTPCheckpoint(checkpointFile);
        await checkpoint.LoadAsync(model, rank, worldSize);

        // Wait for all ranks to finish loading
        // await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Load checkpoint (centralized format) - Module version
    /// </summary>
    public static async Task LoadCentralizedAsync(
        NN.Module model,
        string checkpointFile)
    {
        var rank = 0; // Would get from TensorParallel context
        var worldSize = 1; // Would get from TensorParallel context

        var checkpoint = new CentralizedTPCheckpoint(checkpointFile);
        await checkpoint.LoadAsync(model, rank, worldSize);

        // Wait for all ranks to finish loading
        // await TensorParallel.GetCommunicator().BarrierAsync();
    }

    /// <summary>
    /// Check if checkpoint exists
    /// </summary>
    public static bool CheckpointExists(string checkpointPath, bool isDistributed = true)
    {
        if (isDistributed)
        {
            var metadataFile = Path.Combine(checkpointPath, "metadata.bin");
            return Directory.Exists(checkpointPath) && File.Exists(metadataFile);
        }
        else
        {
            return File.Exists(checkpointPath);
        }
    }

    /// <summary>
    /// Get checkpoint metadata
    /// </summary>
    public static TPCheckpointMetadata? GetMetadata(string checkpointDir)
    {
        var checkpoint = new DistributedTPCheckpoint(checkpointDir);
        return checkpoint.GetMetadata();
    }

    /// <summary>
    /// List available checkpoints in a directory
    /// </summary>
    public static List<string> ListCheckpoints(string checkpointDir)
    {
        var checkpoints = new List<string>();

        if (!Directory.Exists(checkpointDir))
            return checkpoints;

        foreach (var subDir in Directory.GetDirectories(checkpointDir))
        {
            var metadataFile = Path.Combine(subDir, "metadata.bin");
            if (File.Exists(metadataFile))
            {
                checkpoints.Add(Path.GetFileName(subDir));
            }
        }

        return checkpoints;
    }
}
