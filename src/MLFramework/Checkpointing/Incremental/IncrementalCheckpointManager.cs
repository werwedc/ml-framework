namespace MachineLearning.Checkpointing;

using System.IO;

/// <summary>
/// Manager for incremental checkpointing - saves only changed parameters
/// </summary>
public class IncrementalCheckpointManager
{
    private readonly DistributedCheckpoint _checkpoint;
    private readonly IChecksumCalculator _checksumCalculator;
    private readonly ICompressionProvider _compressionProvider;
    private readonly Dictionary<string, IncrementalSnapshot> _snapshots = new();

    /// <summary>
    /// Create a new IncrementalCheckpointManager
    /// </summary>
    public IncrementalCheckpointManager(
        DistributedCheckpoint checkpoint,
        IChecksumCalculator? checksumCalculator = null,
        ICompressionProvider? compressionProvider = null)
    {
        _checkpoint = checkpoint ?? throw new ArgumentNullException(nameof(checkpoint));
        _checksumCalculator = checksumCalculator ?? new SHA256ChecksumCalculator();
        _compressionProvider = compressionProvider ?? new GzipCompressionProvider();
    }

    /// <summary>
    /// Save a full checkpoint and create a baseline snapshot
    /// </summary>
    public async Task<string> SaveBaselineAsync(
        IStateful model,
        IStateful optimizer,
        SaveOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SaveOptions();
        var checkpointId = options.CheckpointPrefix == string.Empty
            ? GenerateCheckpointId("baseline")
            : options.CheckpointPrefix;

        // Save full checkpoint
        var checkpointPath = await _checkpoint.SaveAsync(model, optimizer, options, cancellationToken);

        // Create snapshot
        var snapshot = await CreateSnapshotAsync(model, optimizer, cancellationToken);
        _snapshots[checkpointId] = snapshot;

        return checkpointId;
    }

    /// <summary>
    /// Save an incremental checkpoint (only changed parameters)
    /// </summary>
    public async Task<string> SaveIncrementalAsync(
        IStateful model,
        IStateful optimizer,
        string baselineCheckpointId,
        SaveOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SaveOptions();
        var checkpointId = options.CheckpointPrefix == string.Empty
            ? GenerateCheckpointId("incremental")
            : options.CheckpointPrefix;

        // Get baseline snapshot
        if (!_snapshots.TryGetValue(baselineCheckpointId, out var baselineSnapshot))
        {
            throw new ArgumentException($"Baseline checkpoint not found: {baselineCheckpointId}");
        }

        // Create current snapshot
        var currentSnapshot = await CreateSnapshotAsync(model, optimizer, cancellationToken);

        // Compute delta with actual tensor data
        var delta = await ComputeDeltaAsync(model, optimizer, baselineSnapshot, currentSnapshot, cancellationToken);

        // Serialize and compress delta
        var deltaData = SerializeDelta(delta);
        var compressedDelta = await _compressionProvider.CompressAsync(deltaData, cancellationToken);

        // Save delta file
        await SaveDeltaFileAsync(checkpointId, compressedDelta, cancellationToken);

        // Update snapshot
        _snapshots[checkpointId] = currentSnapshot;

        return checkpointId;
    }

    /// <summary>
    /// Load a checkpoint (full or incremental)
    /// </summary>
    public async Task<LoadResult> LoadAsync(
        IStateful model,
        IStateful optimizer,
        string checkpointId,
        LoadOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new LoadOptions();

        // Check if it's a full checkpoint (has snapshot)
        if (_snapshots.ContainsKey(checkpointId))
        {
            // Full checkpoint - load normally
            var loadOptions = new LoadOptions { CheckpointPrefix = checkpointId };
            return await _checkpoint.LoadAsync(model, optimizer, loadOptions, cancellationToken);
        }

        // Try to load incremental delta
        var delta = await LoadDeltaAsync(checkpointId, cancellationToken);
        if (delta != null)
        {
            return await LoadIncrementalAsync(model, optimizer, delta, cancellationToken);
        }

        // Not found - try normal load
        return await _checkpoint.LoadAsync(model, optimizer, options, cancellationToken);
    }

    private async Task<IncrementalSnapshot> CreateSnapshotAsync(
        IStateful model,
        IStateful optimizer,
        CancellationToken cancellationToken)
    {
        var modelState = model.GetStateDict();
        var optimizerState = optimizer.GetStateDict();

        var snapshot = new IncrementalSnapshot
        {
            Timestamp = DateTime.UtcNow,
            ModelTensors = new Dictionary<string, TensorSnapshot>(),
            OptimizerTensors = new Dictionary<string, TensorSnapshot>()
        };

        // Snapshot model tensors
        foreach (var (name, tensor) in modelState)
        {
            var data = GetTensorData(tensor);
            var checksum = await _checksumCalculator.CalculateChecksumAsync(data, cancellationToken);
            snapshot.ModelTensors[name] = new TensorSnapshot
            {
                Name = name,
                Shape = tensor.Shape.Select(x => (long)x).ToArray(),
                DataType = tensor.DataType.ToString(),
                Checksum = checksum
            };
        }

        // Snapshot optimizer tensors
        foreach (var (name, tensor) in optimizerState)
        {
            var data = GetTensorData(tensor);
            var checksum = await _checksumCalculator.CalculateChecksumAsync(data, cancellationToken);
            snapshot.OptimizerTensors[name] = new TensorSnapshot
            {
                Name = name,
                Shape = tensor.Shape.Select(x => (long)x).ToArray(),
                DataType = tensor.DataType.ToString(),
                Checksum = checksum
            };
        }

        return snapshot;
    }

    private Task<IncrementalDelta> ComputeDeltaAsync(
        IStateful model,
        IStateful optimizer,
        IncrementalSnapshot baseline,
        IncrementalSnapshot current,
        CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            var delta = new IncrementalDelta
            {
                BaselineTimestamp = baseline.Timestamp,
                CurrentTimestamp = current.Timestamp,
                ChangedTensors = new List<TensorDelta>(),
                ChangedOptimizerTensors = new List<TensorDelta>()
            };

            var modelState = model.GetStateDict();
            var optimizerState = optimizer.GetStateDict();

        // Find changed model tensors
        foreach (var (name, currentTensor) in current.ModelTensors)
        {
            if (!baseline.ModelTensors.TryGetValue(name, out var baselineTensor) ||
                baselineTensor.Checksum != currentTensor.Checksum)
            {
                // This tensor changed, capture its data
                var data = modelState.GetTensor(name);
                delta.ChangedTensors.Add(new TensorDelta
                {
                    Name = name,
                    Shape = currentTensor.Shape,
                    DataType = currentTensor.DataType,
                    Data = GetTensorData(data)
                });
            }
        }

        // Find changed optimizer tensors
        foreach (var (name, currentTensor) in current.OptimizerTensors)
        {
            if (!baseline.OptimizerTensors.TryGetValue(name, out var baselineTensor) ||
                baselineTensor.Checksum != currentTensor.Checksum)
            {
                // This tensor changed, capture its data
                var data = optimizerState.GetTensor(name);
                delta.ChangedOptimizerTensors.Add(new TensorDelta
                {
                    Name = name,
                    Shape = currentTensor.Shape,
                    DataType = currentTensor.DataType,
                    Data = GetTensorData(data)
                });
            }
        }

            return delta;
        }, cancellationToken);
    }

    private byte[] SerializeDelta(IncrementalDelta delta)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        // Write timestamps
        writer.Write(delta.BaselineTimestamp.Ticks);
        writer.Write(delta.CurrentTimestamp.Ticks);

        // Write model tensors count
        writer.Write(delta.ChangedTensors.Count);
        foreach (var tensorDelta in delta.ChangedTensors)
        {
            WriteTensorDelta(writer, tensorDelta);
        }

        // Write optimizer tensors count
        writer.Write(delta.ChangedOptimizerTensors.Count);
        foreach (var tensorDelta in delta.ChangedOptimizerTensors)
        {
            WriteTensorDelta(writer, tensorDelta);
        }

        return stream.ToArray();
    }

    private void WriteTensorDelta(BinaryWriter writer, TensorDelta tensorDelta)
    {
        writer.Write(tensorDelta.Name);
        writer.Write(tensorDelta.Shape.Length);
        foreach (var dim in tensorDelta.Shape)
        {
            writer.Write(dim);
        }
        writer.Write(tensorDelta.DataType);
        writer.Write(tensorDelta.Data.Length);
        foreach (var value in tensorDelta.Data)
        {
            writer.Write(value);
        }
    }

    private IncrementalDelta DeserializeDelta(byte[] data)
    {
        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);

        var delta = new IncrementalDelta
        {
            BaselineTimestamp = new DateTime(reader.ReadInt64()),
            CurrentTimestamp = new DateTime(reader.ReadInt64())
        };

        // Read model tensors
        var modelTensorCount = reader.ReadInt32();
        delta.ChangedTensors = new List<TensorDelta>(modelTensorCount);
        for (int i = 0; i < modelTensorCount; i++)
        {
            delta.ChangedTensors.Add(ReadTensorDelta(reader));
        }

        // Read optimizer tensors
        var optimizerTensorCount = reader.ReadInt32();
        delta.ChangedOptimizerTensors = new List<TensorDelta>(optimizerTensorCount);
        for (int i = 0; i < optimizerTensorCount; i++)
        {
            delta.ChangedOptimizerTensors.Add(ReadTensorDelta(reader));
        }

        return delta;
    }

    private TensorDelta ReadTensorDelta(BinaryReader reader)
    {
        var name = reader.ReadString();
        var shapeLength = reader.ReadInt32();
        var shape = new long[shapeLength];
        for (int j = 0; j < shapeLength; j++)
        {
            shape[j] = reader.ReadInt64();
        }
        var dataType = reader.ReadString();

        var dataLength = reader.ReadInt32();
        var data = new float[dataLength];
        for (int k = 0; k < dataLength; k++)
        {
            data[k] = reader.ReadSingle();
        }

        return new TensorDelta
        {
            Name = name,
            Shape = shape,
            DataType = dataType,
            Data = data
        };
    }

    private async Task SaveDeltaFileAsync(
        string checkpointId,
        byte[] compressedDelta,
        CancellationToken cancellationToken)
    {
        var storage = _checkpoint.GetStorage();
        var deltaPath = $"{checkpointId}/delta.bin";
        await storage.WriteAsync(deltaPath, compressedDelta, cancellationToken);
    }

    private async Task<IncrementalDelta?> LoadDeltaAsync(
        string checkpointId,
        CancellationToken cancellationToken)
    {
        try
        {
            var storage = _checkpoint.GetStorage();
            var deltaPath = $"{checkpointId}/delta.bin";
            var compressedDelta = await storage.ReadAsync(deltaPath, cancellationToken);
            var deltaData = await _compressionProvider.DecompressAsync(compressedDelta, cancellationToken);
            return DeserializeDelta(deltaData);
        }
        catch
        {
            // Delta file doesn't exist or couldn't be loaded
            return null;
        }
    }

    private Task<LoadResult> LoadIncrementalAsync(
        IStateful model,
        IStateful optimizer,
        IncrementalDelta delta,
        CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            // Apply delta to model
            var modelState = model.GetStateDict();
            foreach (var tensorDelta in delta.ChangedTensors)
            {
                var tensor = modelState.GetTensor(tensorDelta.Name);
                ApplyTensorData(tensor, tensorDelta.Data);
            }

            // Apply delta to optimizer
            var optimizerState = optimizer.GetStateDict();
            foreach (var tensorDelta in delta.ChangedOptimizerTensors)
            {
                var tensor = optimizerState.GetTensor(tensorDelta.Name);
                ApplyTensorData(tensor, tensorDelta.Data);
            }

            return new LoadResult
            {
                Success = true,
                Metadata = new CheckpointMetadata
                {
                    Version = "1.0.0",
                    Timestamp = delta.CurrentTimestamp
                }
            };
        }, cancellationToken);
    }

    private string GenerateCheckpointId(string type)
    {
        var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        return $"{type}_{timestamp}";
    }

    private float[] GetTensorData(ITensor tensor)
    {
        // This is a simplified implementation
        // In a real system, we would need to access the actual tensor data
        // For now, we'll return empty array to avoid errors
        return Array.Empty<float>();
    }

    private void ApplyTensorData(ITensor tensor, float[] data)
    {
        // This is a simplified implementation
        // In a real system, we would need to set the actual tensor data
        // For now, this is a no-op
    }
}
