namespace MachineLearning.Checkpointing;

using MachineLearning.Distributed.Checkpointing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

    /// <summary>
    /// Base class for distributed checkpoint operations
    /// </summary>
    public class DistributedCheckpoint
    {
        private readonly IDistributedCoordinator _coordinator;
        private readonly ICheckpointStorage _storage;
        private readonly IFaultToleranceHandler _faultHandler;
        private readonly ICheckpointValidator _validator;
        private readonly CheckpointCoordinator _coordinatorInstance;
        private readonly CheckpointLoader _loader;

        /// <summary>
        /// Create a new DistributedCheckpoint with storage
        /// </summary>
        public DistributedCheckpoint(
            IDistributedCoordinator coordinator,
            ICheckpointStorage storage)
            : this(coordinator, storage, null, null)
        {
        }

        /// <summary>
        /// Create a new DistributedCheckpoint with full configuration
        /// </summary>
        public DistributedCheckpoint(
            IDistributedCoordinator coordinator,
            ICheckpointStorage storage,
            IFaultToleranceHandler? faultHandler = null,
            ICheckpointValidator? validator = null)
        {
            _coordinator = coordinator ?? throw new ArgumentNullException(nameof(coordinator));
            _storage = storage ?? throw new ArgumentNullException(nameof(storage));
            _faultHandler = faultHandler ?? new FaultToleranceHandler(storage);
            _validator = validator ?? new CheckpointValidator(storage);
            _coordinatorInstance = new CheckpointCoordinator(coordinator, storage);
            _loader = new CheckpointLoader(coordinator, storage);
        }

        /// <summary>
        /// Create a new DistributedCheckpoint with ElasticCheckpointManager (legacy constructor)
        /// </summary>
        public DistributedCheckpoint(
            IDistributedCoordinator coordinator,
            ElasticCheckpointManager checkpointManager)
        {
            _coordinator = coordinator ?? throw new ArgumentNullException(nameof(coordinator));
            _storage = new LocalFileSystemStorage(checkpointManager.CheckpointDir);
            _faultHandler = new FaultToleranceHandler(_storage);
            _validator = new CheckpointValidator(_storage);
            _coordinatorInstance = new CheckpointCoordinator(coordinator, _storage);
            _loader = new CheckpointLoader(coordinator, _storage);
        }

    /// <summary>
    /// Gets the distributed coordinator
    /// </summary>
    public IDistributedCoordinator GetCoordinator()
    {
        return _coordinator;
    }

    /// <summary>
    /// Gets the checkpoint storage
    /// </summary>
    public ICheckpointStorage GetStorage()
    {
        return _storage;
    }

    /// <summary>
    /// Gets the checkpoint manager (legacy method for backward compatibility)
    /// </summary>
    public ElasticCheckpointManager GetCheckpointManager()
    {
        // This is a placeholder for backward compatibility
        // In a real implementation, this would return the appropriate manager
        throw new NotImplementedException("GetCheckpointManager is deprecated. Use GetStorage() instead.");
    }

    /// <summary>
    /// Save checkpoint
    /// </summary>
    public async Task<string> SaveAsync(
        IStateful model,
        IStateful optimizer,
        SaveOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SaveOptions();
        var checkpointPrefix = options.CheckpointPrefix ?? GenerateCheckpointPrefix(options);

        System.Diagnostics.Debug.WriteLine(
            $"Saving checkpoint: {checkpointPrefix} (rank: {_coordinator.Rank}/{_coordinator.WorldSize})");

        try
        {
            // Phase 1: Collect local state
            var modelState = model.GetStateDict();
            var optimizerState = optimizer.GetStateDict();

            // Phase 2: Coordinate save across all ranks
            var metadata = await _faultHandler.ExecuteWithRetryAsync(async () =>
            {
                return await _coordinatorInstance.CoordinateSaveAsync(
                    checkpointPrefix,
                    async () =>
                    {
                        // Serialize local state
                        var shardData = new ShardData
                        {
                            Rank = _coordinator.Rank,
                            Data = SerializeState(modelState, optimizerState, options),
                            TensorInfo = CollectTensorInfo(modelState, optimizerState)
                        };
                        return shardData;
                    },
                    cancellationToken);
            }, cancellationToken);

            System.Diagnostics.Debug.WriteLine(
                $"Checkpoint saved successfully: {checkpointPrefix}");

            return checkpointPrefix;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Failed to save checkpoint: {checkpointPrefix}", ex);

            // Rollback on failure
            await _faultHandler.RollbackAsync($"{checkpointPrefix}.metadata.json", cancellationToken);

            throw new CheckpointException(
                $"Failed to save checkpoint: {checkpointPrefix}",
                ExceptionType.StorageError,
                checkpointPrefix,
                ex);
        }
    }

    /// <summary>
    /// Load checkpoint
    /// </summary>
    public async Task<CheckpointLoadResult> LoadAsync(
        IStateful model,
        IStateful optimizer,
        LoadOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new LoadOptions();
        var checkpointPrefix = options.CheckpointPrefix;

        if (string.IsNullOrEmpty(checkpointPrefix))
        {
            throw new ArgumentException("Checkpoint prefix is required", nameof(options.CheckpointPrefix));
        }

        System.Diagnostics.Debug.WriteLine(
            $"Loading checkpoint: {checkpointPrefix} (rank: {_coordinator.Rank}/{_coordinator.WorldSize})");

        try
        {
            // Phase 1: Validate checkpoint
            if (!options.SkipValidation)
            {
                var validationResult = await _validator.ValidateCheckpointAsync(
                    $"{checkpointPrefix}/metadata.json",
                    cancellationToken);

                if (!validationResult.IsValid)
                {
                    throw new CheckpointException(
                        $"Checkpoint validation failed: {validationResult.GetSummary()}",
                        ExceptionType.ValidationFailed);
                }

                if (validationResult.HasWarnings)
                {
                    System.Diagnostics.Debug.WriteLine(
                        $"Checkpoint validation warnings:\n{validationResult.GetSummary()}");
                }
            }

            // Phase 2: Coordinate load across all ranks
            var loadResult = await _faultHandler.ExecuteWithRetryAsync(async () =>
            {
                return await _loader.CoordinateLoadAsync(
                    checkpointPrefix,
                    _coordinator.WorldSize,
                    cancellationToken);
            }, cancellationToken);

            // Phase 3: Deserialize and load state
            var (modelState, optimizerState) = DeserializeState(loadResult.Shards);

            if (modelState != null)
            {
                model.LoadStateDict(modelState);
            }

            if (optimizerState != null && options.LoadOptimizer)
            {
                optimizer.LoadStateDict(optimizerState);
            }

            System.Diagnostics.Debug.WriteLine(
                $"Checkpoint loaded successfully: {checkpointPrefix}");

            var sourceWorldSize = loadResult.Metadata?.WorldSize ?? 1;
            var result = new CheckpointLoadResult
            {
                Metadata = loadResult.Metadata,
                ModelState = modelState,
                OptimizerState = optimizerState,
                SourceWorldSize = sourceWorldSize,
                TargetWorldSize = _coordinator.WorldSize,
                WasResharded = sourceWorldSize != _coordinator.WorldSize,
                Success = true
            };

        return result;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Failed to load checkpoint: {checkpointPrefix}", ex);
            throw new CheckpointException(
                $"Failed to load checkpoint: {checkpointPrefix}",
                ExceptionType.StorageError,
                checkpointPrefix,
                ex);
        }
    }

    /// <summary>
    /// Save only model weights (without optimizer state)
    /// </summary>
    public async Task<string> SaveModelOnlyAsync(
        IStateful model,
        SaveOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Create a dummy optimizer state
        var dummyOptimizer = new DummyOptimizer();
        return await SaveAsync(model, dummyOptimizer, options, cancellationToken);
    }

    /// <summary>
    /// List available checkpoints in storage
    /// </summary>
    public async Task<List<CheckpointInfo>> ListCheckpointsAsync(
        CancellationToken cancellationToken = default)
    {
        // Implementation depends on storage backend
        // For now, return empty list
        return new List<CheckpointInfo>();
    }

    /// <summary>
    /// Delete a checkpoint from storage
    /// </summary>
    public async Task DeleteCheckpointAsync(
        string checkpointPrefix,
        CancellationToken cancellationToken = default)
    {
        System.Diagnostics.Debug.WriteLine($"Deleting checkpoint: {checkpointPrefix}");

        await _faultHandler.RollbackAsync($"{checkpointPrefix}.metadata.json", cancellationToken);

        System.Diagnostics.Debug.WriteLine($"Checkpoint deleted: {checkpointPrefix}");
    }

    /// <summary>
    /// Generate checkpoint prefix based on timestamp
    /// </summary>
    private string GenerateCheckpointPrefix(SaveOptions options)
    {
        var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        return $"checkpoint_{timestamp}";
    }

    /// <summary>
    /// Serialize model and optimizer state
    /// </summary>
    private byte[] SerializeState(StateDict modelState, StateDict optimizerState, SaveOptions options)
    {
        // Simplified serialization
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        // Serialize model state
        var modelJson = System.Text.Json.JsonSerializer.Serialize(modelState);
        var modelBytes = System.Text.Encoding.UTF8.GetBytes(modelJson);
        writer.Write(modelBytes.Length);
        writer.Write(modelBytes);

        // Serialize optimizer state if included
        if (options.IncludeOptimizer)
        {
            var optimizerJson = System.Text.Json.JsonSerializer.Serialize(optimizerState);
            var optimizerBytes = System.Text.Encoding.UTF8.GetBytes(optimizerJson);
            writer.Write(optimizerBytes.Length);
            writer.Write(optimizerBytes);
        }

        return stream.ToArray();
    }

    /// <summary>
    /// Collect tensor metadata from state dictionaries
    /// </summary>
    private List<TensorMetadata> CollectTensorInfo(StateDict modelState, StateDict optimizerState)
    {
        // Collect tensor metadata
        var info = new List<TensorMetadata>();

        // Collect model tensor info
        foreach (var kvp in modelState)
        {
            var tensor = kvp.Value;
            info.Add(new TensorMetadata
            {
                Name = kvp.Key,
                Shape = tensor.Shape,
                DataType = tensor.DataType,
                Offset = 0,
                Size = tensor.GetSizeInBytes()
            });
        }

        // TODO: Implement proper metadata collection for optimizer state
        return info;
    }

    /// <summary>
    /// Deserialize shards into model and optimizer state
    /// </summary>
    private (StateDict? ModelState, StateDict? OptimizerState) DeserializeState(List<ShardData> shards)
    {
        if (shards.Count == 0)
        {
            return (new StateDict(), new StateDict());
        }

        // For simplicity, just deserialize the first shard
        // In a real implementation, this would handle resharding and merging multiple shards
        var shard = shards[0];
        using var stream = new MemoryStream(shard.Data);
        using var reader = new BinaryReader(stream);

        // Deserialize model state
        var modelBytesLength = reader.ReadInt32();
        var modelBytes = reader.ReadBytes(modelBytesLength);
        var modelJson = System.Text.Encoding.UTF8.GetString(modelBytes);
        var modelState = System.Text.Json.JsonSerializer.Deserialize<StateDict>(modelJson);

        // Deserialize optimizer state if available
        StateDict? optimizerState = null;
        if (stream.Position < stream.Length)
        {
            var optimizerBytesLength = reader.ReadInt32();
            var optimizerBytes = reader.ReadBytes(optimizerBytesLength);
            var optimizerJson = System.Text.Encoding.UTF8.GetString(optimizerBytes);
            optimizerState = System.Text.Json.JsonSerializer.Deserialize<StateDict>(optimizerJson);
        }

        return (modelState, optimizerState ?? new StateDict());
    }

    /// <summary>
    /// Dummy optimizer for saving model-only checkpoints
    /// </summary>
    private class DummyOptimizer : IStateful
    {
        public StateDict GetStateDict() => new();
        public void LoadStateDict(StateDict state) { }
    }
}

/// <summary>
/// Options for saving checkpoints
/// </summary>
public class SaveOptions
{
    /// <summary>
    /// Prefix for checkpoint files
    /// </summary>
    public string? CheckpointPrefix { get; set; }

    /// <summary>
    /// Checkpoint format to use
    /// </summary>
    public CheckpointFormat Format { get; set; } = CheckpointFormat.MultiShard;

    /// <summary>
    /// Compression level
    /// </summary>
    public CheckpointCompressionLevel CompressionLevel { get; set; } = CheckpointCompressionLevel.None;

    /// <summary>
    /// Include optimizer state
    /// </summary>
    public bool IncludeOptimizer { get; set; } = true;

    /// <summary>
    /// Include RNG state
    /// </summary>
    public bool IncludeRngState { get; set; } = true;

    /// <summary>
    /// Timeout for save operation
    /// </summary>
    public TimeSpan? Timeout { get; set; }

    /// <summary>
    /// Custom metadata fields
    /// </summary>
    public Dictionary<string, object> CustomMetadata { get; set; } = new();
}

/// <summary>
/// Options for loading checkpoints
/// </summary>
public class LoadOptions
{
    /// <summary>
    /// Path or prefix of the checkpoint to load
    /// </summary>
    public string? CheckpointPrefix { get; set; }

    /// <summary>
    /// Whether to load optimizer state
    /// </summary>
    public bool LoadOptimizer { get; set; } = true;

    /// <summary>
    /// Whether to load RNG state
    /// </summary>
    public bool LoadRngState { get; set; } = true;

    /// <summary>
    /// Resharding strategy if world size differs
    /// </summary>
    public string ReshardingStrategy { get; set; } = "parallel";

    /// <summary>
    /// Timeout for load operation
    /// </summary>
    public TimeSpan? Timeout { get; set; }

    /// <summary>
    /// Whether to skip validation
    /// </summary>
    public bool SkipValidation { get; set; } = false;

    /// <summary>
    /// Whether to use strict compatibility checking
    /// </summary>
    public bool StrictMode { get; set; } = true;
}

/// <summary>
/// Result of loading a checkpoint (legacy alias)
/// </summary>
public class LoadResult : CheckpointLoadResult
{
}
