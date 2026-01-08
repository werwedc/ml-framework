namespace MachineLearning.Distributed.Checkpointing;

using MachineLearning.Distributed.Models;

/// <summary>
/// Manages checkpointing for elastic training with support for rescaling
/// </summary>
public class ElasticCheckpointManager
{
    private readonly string _checkpointDir;
    private readonly int _maxCheckpointsToKeep;
    private readonly Dictionary<string, TrainingCheckpoint> _checkpointCache;

    /// <summary>
    /// Gets the directory where checkpoints are stored
    /// </summary>
    public string CheckpointDir => _checkpointDir;

    public ElasticCheckpointManager(
        string checkpointDir,
        int maxCheckpointsToKeep = 3)
    {
        if (string.IsNullOrWhiteSpace(checkpointDir))
            throw new ArgumentException("Checkpoint directory cannot be empty", nameof(checkpointDir));

        if (maxCheckpointsToKeep <= 0)
            throw new ArgumentException("Max checkpoints to keep must be positive", nameof(maxCheckpointsToKeep));

        _checkpointDir = checkpointDir;
        _maxCheckpointsToKeep = maxCheckpointsToKeep;
        _checkpointCache = new Dictionary<string, TrainingCheckpoint>();

        // Ensure checkpoint directory exists
        Directory.CreateDirectory(_checkpointDir);
    }

    /// <summary>
    /// Save a full checkpoint before rescaling
    /// </summary>
    public async Task SaveCheckpointAsync(
        TrainingCheckpoint checkpoint,
        bool isRescalingCheckpoint = false)
    {
        if (checkpoint == null)
            throw new ArgumentNullException(nameof(checkpoint));

        checkpoint.Timestamp = DateTime.UtcNow;
        checkpoint.IsRescalingCheckpoint = isRescalingCheckpoint;

        var filename = GetCheckpointFilename(checkpoint);
        var filepath = Path.Combine(_checkpointDir, filename);

        // Save to disk
        await SaveCheckpointToFileAsync(checkpoint, filepath);

        // Update cache
        _checkpointCache[filename] = checkpoint;

        // Cleanup old checkpoints
        CleanupOldCheckpoints();
    }

    /// <summary>
    /// Load the latest checkpoint
    /// </summary>
    public async Task<TrainingCheckpoint?> LoadLatestCheckpointAsync()
    {
        var checkpoints = ListAvailableCheckpoints();
        if (checkpoints.Count == 0)
        {
            return null;
        }

        var latest = checkpoints.OrderByDescending(c => c.Timestamp).First();
        return await LoadCheckpointAsync(latest.Id);
    }

    /// <summary>
    /// Load a specific checkpoint by ID
    /// </summary>
    public async Task<TrainingCheckpoint?> LoadCheckpointAsync(string checkpointId)
    {
        if (string.IsNullOrWhiteSpace(checkpointId))
            throw new ArgumentException("Checkpoint ID cannot be empty", nameof(checkpointId));

        var filepath = Path.Combine(_checkpointDir, $"{checkpointId}.ckpt");

        if (!File.Exists(filepath))
        {
            return null;
        }

        // Check cache first
        if (_checkpointCache.TryGetValue($"{checkpointId}.ckpt", out var cachedCheckpoint))
        {
            return cachedCheckpoint;
        }

        // Load from disk
        var checkpoint = await LoadCheckpointFromFileAsync(filepath);

        // Add to cache
        _checkpointCache[$"{checkpointId}.ckpt"] = checkpoint;

        return checkpoint;
    }

    /// <summary>
    /// Create a checkpoint for rescaling (minimal state)
    /// </summary>
    public TrainingCheckpoint CreateRescalingCheckpoint(
        GlobalTrainingState state,
        byte[]? modelState = null,
        byte[]? optimizerState = null)
    {
        return new TrainingCheckpoint
        {
            Id = $"rescaling_{Guid.NewGuid():N}",
            Epoch = state.CurrentEpoch,
            Step = state.CurrentStep,
            LearningRate = state.LearningRate,
            ModelState = modelState,
            OptimizerState = optimizerState,
            WorkerCount = state.ActiveWorkerCount,
            IsRescalingCheckpoint = true,
            Timestamp = DateTime.UtcNow
        };
    }

    /// <summary>
    /// List all available checkpoints
    /// </summary>
    public List<TrainingCheckpoint> ListAvailableCheckpoints()
    {
        var checkpoints = new List<TrainingCheckpoint>();

        if (!Directory.Exists(_checkpointDir))
        {
            return checkpoints;
        }

        var files = Directory.GetFiles(_checkpointDir, "*.ckpt");
        foreach (var file in files)
        {
            try
            {
                var checkpoint = LoadCheckpointFromFileAsync(file).GetAwaiter().GetResult();
                if (checkpoint != null)
                {
                    checkpoints.Add(checkpoint);
                }
            }
            catch
            {
                // Skip corrupted checkpoint files
            }
        }

        return checkpoints.OrderByDescending(c => c.Timestamp).ToList();
    }

    /// <summary>
    /// Delete a specific checkpoint
    /// </summary>
    public void DeleteCheckpoint(string checkpointId)
    {
        if (string.IsNullOrWhiteSpace(checkpointId))
            throw new ArgumentException("Checkpoint ID cannot be empty", nameof(checkpointId));

        var filename = $"{checkpointId}.ckpt";
        var filepath = Path.Combine(_checkpointDir, filename);

        if (File.Exists(filepath))
        {
            File.Delete(filepath);
        }

        _checkpointCache.Remove(filename);
    }

    /// <summary>
    /// Delete all checkpoints
    /// </summary>
    public void DeleteAllCheckpoints()
    {
        if (!Directory.Exists(_checkpointDir))
        {
            return;
        }

        var files = Directory.GetFiles(_checkpointDir, "*.ckpt");
        foreach (var file in files)
        {
            File.Delete(file);
        }

        _checkpointCache.Clear();
    }

    /// <summary>
    /// Validate checkpoint integrity
    /// </summary>
    public bool ValidateCheckpoint(TrainingCheckpoint checkpoint)
    {
        if (checkpoint == null)
            return false;

        if (string.IsNullOrWhiteSpace(checkpoint.Id))
            return false;

        if (checkpoint.Epoch < 0)
            return false;

        if (checkpoint.Step < 0)
            return false;

        if (checkpoint.WorkerCount <= 0)
            return false;

        return true;
    }

    private async Task SaveCheckpointToFileAsync(TrainingCheckpoint checkpoint, string filepath)
    {
        // In a full implementation, this would serialize the checkpoint to a file format
        // For now, we'll use a simple JSON-based serialization

        var json = System.Text.Json.JsonSerializer.Serialize(checkpoint);
        await File.WriteAllTextAsync(filepath, json);
    }

    private async Task<TrainingCheckpoint> LoadCheckpointFromFileAsync(string filepath)
    {
        // In a full implementation, this would deserialize from the checkpoint file format
        // For now, we'll use simple JSON deserialization

        var json = await File.ReadAllTextAsync(filepath);
        return System.Text.Json.JsonSerializer.Deserialize<TrainingCheckpoint>(json)
            ?? throw new InvalidOperationException("Failed to deserialize checkpoint");
    }

    private string GetCheckpointFilename(TrainingCheckpoint checkpoint)
    {
        return $"{checkpoint.Id}.ckpt";
    }

    private void CleanupOldCheckpoints()
    {
        var checkpoints = ListAvailableCheckpoints();

        // Always keep the latest rescaling checkpoint
        var rescalingCheckpoints = checkpoints
            .Where(c => c.IsRescalingCheckpoint)
            .OrderByDescending(c => c.Timestamp)
            .Take(1);

        // Keep top N regular checkpoints
        var regularCheckpoints = checkpoints
            .Where(c => !c.IsRescalingCheckpoint)
            .OrderByDescending(c => c.Timestamp)
            .Take(_maxCheckpointsToKeep);

        var toKeep = rescalingCheckpoints.Union(regularCheckpoints).ToList();
        var toDelete = checkpoints.Except(toKeep).ToList();

        foreach (var checkpoint in toDelete)
        {
            DeleteCheckpoint(checkpoint.Id);
        }
    }
}
