# Spec: Distributed Checkpoint API

## Overview
Define the main API surface for distributed checkpointing, providing a clean, user-friendly interface for saving and loading checkpoints across multiple GPUs.

## Scope
- 45-60 minutes coding time
- Focus on API design and integration
- Target: `src/MLFramework/Checkpointing/`

## Classes

### 1. DistributedCheckpoint (Main API Class)
```csharp
public class DistributedCheckpoint
{
    private readonly IDistributedCoordinator _coordinator;
    private readonly ICheckpointStorage _storage;
    private readonly IFaultToleranceHandler _faultHandler;
    private readonly ICheckpointValidator _validator;
    private readonly CheckpointCoordinator _coordinator;
    private readonly CheckpointLoader _loader;
    private readonly ILogger<DistributedCheckpoint> _logger;

    public DistributedCheckpoint(
        IDistributedCoordinator coordinator,
        ICheckpointStorage? storage = null,
        IFaultToleranceHandler? faultHandler = null,
        ICheckpointValidator? validator = null,
        ILogger<DistributedCheckpoint>? logger = null)
    {
        _coordinator = coordinator;
        _storage = storage ?? StorageFactory.Create(new StorageOptions
        {
            Provider = "local",
            ConnectionSettings = new Dictionary<string, string>
            {
                ["basePath"] = "./checkpoints"
            }
        });
        _faultHandler = faultHandler ?? new FaultToleranceHandler(_storage);
        _validator = validator ?? new CheckpointValidator(_storage);
        _coordinator = new CheckpointCoordinator(_coordinator, _storage);
        _loader = new CheckpointLoader(_coordinator, _storage);
        _logger = logger;
    }

    /// <summary>
    /// Save a checkpoint (model + optimizer) with optional additional state
    /// </summary>
    public async Task<string> SaveAsync(
        IStateful model,
        IStateful optimizer,
        SaveOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SaveOptions();
        var checkpointPrefix = options.CheckpointPrefix ?? GenerateCheckpointPrefix(options);

        _logger?.LogInformation(
            "Saving checkpoint: {CheckpointPrefix} (rank: {Rank}/{WorldSize})",
            checkpointPrefix,
            _coordinator.Rank,
            _coordinator.WorldSize);

        try
        {
            // Phase 1: Collect local state
            var modelState = model.GetStateDict();
            var optimizerState = optimizer.GetStateDict();

            // Phase 2: Coordinate save across all ranks
            var metadata = await _faultHandler.ExecuteWithRetryAsync(async () =>
            {
                return await _coordinator.CoordinateSaveAsync(
                    checkpointPrefix,
                    async () =>
                    {
                        // Serialize local state
                        var shardData = new ShardData
                        {
                            Data = SerializeState(modelState, optimizerState),
                            TensorInfo = CollectTensorInfo(modelState, optimizerState)
                        };
                        return shardData;
                    },
                    cancellationToken);
            }, cancellationToken);

            _logger?.LogInformation(
                "Checkpoint saved successfully: {CheckpointPrefix}",
                checkpointPrefix);

            return checkpointPrefix;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Failed to save checkpoint: {CheckpointPrefix}", checkpointPrefix);

            // Rollback on failure
            await _faultHandler.RollbackAsync($"{checkpointPrefix}.metadata.json", cancellationToken);

            throw new CheckpointException($"Failed to save checkpoint: {checkpointPrefix}", ex);
        }
    }

    /// <summary>
    /// Load a checkpoint (model + optimizer) with optional resharding
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

        _logger?.LogInformation(
            "Loading checkpoint: {CheckpointPrefix} (rank: {Rank}/{WorldSize})",
            checkpointPrefix,
            _coordinator.Rank,
            _coordinator.WorldSize);

        try
        {
            // Phase 1: Validate checkpoint
            var validationResult = await _validator.ValidateCheckpointAsync(
                $"{checkpointPrefix}.metadata.json",
                cancellationToken);

            if (!validationResult.IsValid)
            {
                throw new CheckpointException(
                    $"Checkpoint validation failed: {validationResult.GetSummary()}");
            }

            if (validationResult.HasWarnings)
            {
                _logger?.LogWarning(
                    "Checkpoint validation warnings:\n{Warnings}",
                    validationResult.GetSummary());
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
            model.LoadStateDict(modelState);
            optimizer.LoadStateDict(optimizerState);

            _logger?.LogInformation(
                "Checkpoint loaded successfully: {CheckpointPrefix}",
                checkpointPrefix);

            return new CheckpointLoadResult
            {
                Metadata = loadResult.Metadata,
                ModelState = modelState,
                OptimizerState = optimizerState
            };
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Failed to load checkpoint: {CheckpointPrefix}", checkpointPrefix);
            throw new CheckpointException($"Failed to load checkpoint: {checkpointPrefix}", ex);
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
        _logger?.LogInformation("Deleting checkpoint: {CheckpointPrefix}", checkpointPrefix);

        await _faultHandler.RollbackAsync($"{checkpointPrefix}.metadata.json", cancellationToken);

        _logger?.LogInformation("Checkpoint deleted: {CheckpointPrefix}", checkpointPrefix);
    }

    private string GenerateCheckpointPrefix(SaveOptions options)
    {
        var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        return $"checkpoint_{timestamp}";
    }

    private byte[] SerializeState(StateDict modelState, StateDict optimizerState)
    {
        // Simplified serialization
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        // TODO: Implement proper serialization
        return stream.ToArray();
    }

    private List<TensorMetadata> CollectTensorInfo(StateDict modelState, StateDict optimizerState)
    {
        // Collect tensor metadata
        var info = new List<TensorMetadata>();
        // TODO: Implement proper metadata collection
        return info;
    }

    private (StateDict ModelState, StateDict OptimizerState) DeserializeState(List<ShardData> shards)
    {
        // Simplified deserialization
        return (new StateDict(), new StateDict());
    }

    private class DummyOptimizer : IStateful
    {
        public StateDict GetStateDict() => new();
        public void LoadStateDict(StateDict state) { }
    }
}
```

### 2. SaveOptions (Save Configuration)
```csharp
public class SaveOptions
{
    /// <summary>
    /// Custom checkpoint prefix (optional)
    /// </summary>
    public string? CheckpointPrefix { get; set; }

    /// <summary>
    /// Checkpoint format to use
    /// </summary>
    public CheckpointFormat Format { get; set; } = CheckpointFormat.MultiShard;

    /// <summary>
    /// Compression level
    /// </summary>
    public CompressionLevel CompressionLevel { get; set; } = CompressionLevel.None;

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

public enum CheckpointFormat
{
    SingleFile,
    MultiShard,
    Consolidated
}

public enum CompressionLevel
{
    None,
    Fast,
    Balanced,
    Maximum
}
```

### 3. LoadOptions (Load Configuration)
```csharp
public class LoadOptions
{
    /// <summary>
    /// Checkpoint prefix to load
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
```

### 4. CheckpointLoadResult (Load Result)
```csharp
public class CheckpointLoadResult
{
    public CheckpointMetadata Metadata { get; set; }
    public StateDict ModelState { get; set; }
    public StateDict OptimizerState { get; set; }
    public int SourceWorldSize { get; set; }
    public int TargetWorldSize { get; set; }
    public bool WasResharded { get; set; }
}
```

### 5. CheckpointInfo (Checkpoint Information)
```csharp
public class CheckpointInfo
{
    public string Prefix { get; set; }
    public string Version { get; set; }
    public DateTime Timestamp { get; set; }
    public int WorldSize { get; set; }
    public long SizeBytes { get; set; }
    public string Format { get; set; }
}
```

### 6. CheckpointException (Checkpoint-Specific Exception)
```csharp
public class CheckpointException : Exception
{
    public string CheckpointPath { get; }
    public ExceptionType Type { get; }

    public CheckpointException(
        string message,
        string checkpointPath = "",
        ExceptionType type = ExceptionType.Unknown,
        Exception? innerException = null)
        : base(message, innerException)
    {
        CheckpointPath = checkpointPath;
        Type = type;
    }

    public CheckpointException(
        string message,
        Exception? innerException)
        : base(message, innerException)
    {
        Type = ExceptionType.Unknown;
    }
}

public enum ExceptionType
{
    Unknown,
    SaveFailed,
    LoadFailed,
    ValidationFailed,
    CorruptedCheckpoint,
    IncompatibleVersion,
    Timeout,
    StorageError
}
```

### 7. DistributedCheckpointFactory (Factory for Creating Checkpoint Instances)
```csharp
public static class DistributedCheckpointFactory
{
    public static DistributedCheckpoint Create(
        IDistributedCoordinator coordinator,
        CheckpointOptions? options = null)
    {
        options ??= new CheckpointOptions();

        // Create storage
        var storage = StorageFactory.Create(options.Storage);

        // Create fault handler
        var faultHandler = new FaultToleranceHandler(
            storage,
            options.RetryPolicy);

        // Create validator
        var validator = new CheckpointValidator(
            storage,
            options.IntegrityCheckers,
            options.CompatibilityCheckers);

        return new DistributedCheckpoint(
            coordinator,
            storage,
            faultHandler,
            validator);
    }
}
```

### 8. CheckpointOptions (Global Configuration)
```csharp
public class CheckpointOptions
{
    public StorageOptions Storage { get; set; } = new();
    public RetryPolicy RetryPolicy { get; set; } = new();
    public List<IIntegrityChecker> IntegrityCheckers { get; set; } = new();
    public List<ICompatibilityChecker> CompatibilityCheckers { get; set} = new();
    public TimeSpan DefaultTimeout { get; set; } = TimeSpan.FromMinutes(10);
}
```

## Integration Points
- Uses: All checkpointing components (Coordinator, Loader, Validator, etc.)
- Depends on: `IDistributedCoordinator`, `ICheckpointStorage`, `IStateful`

## Usage Examples

### Saving a Checkpoint
```csharp
var checkpoint = DistributedCheckpointFactory.Create(coordinator);
await checkpoint.SaveAsync(model, optimizer, new SaveOptions
{
    CheckpointPrefix = "model_epoch_100",
    Format = CheckpointFormat.MultiShard,
    IncludeRngState = true
});
```

### Loading a Checkpoint
```csharp
var checkpoint = DistributedCheckpointFactory.Create(coordinator);
var result = await checkpoint.LoadAsync(model, optimizer, new LoadOptions
{
    CheckpointPrefix = "model_epoch_100",
    ReshardingStrategy = "parallel"
});
```

### Listing Checkpoints
```csharp
var checkpoint = DistributedCheckpointFactory.Create(coordinator);
var checkpoints = await checkpoint.ListCheckpointsAsync();
foreach (var info in checkpoints)
{
    Console.WriteLine($"{info.Prefix} - {info.Timestamp} - {info.WorldSize} GPUs");
}
```

## Testing Requirements
- Test save/load roundtrip
- Test with different options
- Test error handling and exceptions
- Test checkpoint listing
- Test checkpoint deletion
- Test timeout handling
- Test resharding scenarios

## Success Criteria
- Clean, intuitive API surface
- Consistent naming conventions
- Comprehensive error handling
- Flexible configuration options
- Well-documented usage
- Support for common checkpointing scenarios
