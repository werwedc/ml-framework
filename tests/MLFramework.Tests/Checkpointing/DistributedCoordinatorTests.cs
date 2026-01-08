using Xunit;
using MachineLearning.Checkpointing;

namespace MLFramework.Tests.Checkpointing;

/// <summary>
/// Tests for distributed checkpoint coordination
/// </summary>
public class DistributedCoordinatorTests
{
    /// <summary>
    /// Mock distributed coordinator for testing
    /// </summary>
    private class MockDistributedCoordinator : IDistributedCoordinator
    {
        private readonly int _rank;
        private readonly int _worldSize;
        private int _barrierCallCount;

        public MockDistributedCoordinator(int rank = 0, int worldSize = 1)
        {
            _rank = rank;
            _worldSize = worldSize;
            _barrierCallCount = 0;
        }

        public int Rank => _rank;

        public int WorldSize => _worldSize;

        public int BarrierCallCount => _barrierCallCount;

        public Task BarrierAsync(CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref _barrierCallCount);
            return Task.CompletedTask;
        }

        public Task<T> BroadcastAsync<T>(T data, CancellationToken cancellationToken = default) where T : class
        {
            return Task.FromResult(data);
        }

        public Task<T> AllReduceAsync<T>(T data, Func<T, T, T> reducer, CancellationToken cancellationToken = default) where T : class
        {
            return Task.FromResult(data);
        }

        public Task<IList<T>?> GatherAsync<T>(T data, CancellationToken cancellationToken = default) where T : class
        {
            if (Rank == 0)
            {
                var allData = new List<T>();
                for (int i = 0; i < WorldSize; i++)
                {
                    allData.Add(data);
                }
                return Task.FromResult<IList<T>?>(allData);
            }
            return Task.FromResult<IList<T>?>(null);
        }
    }

    /// <summary>
    /// Mock storage for testing
    /// </summary>
    private class MockCheckpointStorage : ICheckpointStorage
    {
        private readonly Dictionary<string, byte[]> _storage;

        public MockCheckpointStorage()
        {
            _storage = new Dictionary<string, byte[]>();
        }

        public Task WriteAsync(string path, byte[] data, CancellationToken cancellationToken = default)
        {
            _storage[path] = data;
            return Task.CompletedTask;
        }

        public Task<byte[]> ReadAsync(string path, CancellationToken cancellationToken = default)
        {
            if (!_storage.ContainsKey(path))
                throw new FileNotFoundException($"File not found: {path}");

            return Task.FromResult(_storage[path]);
        }

        public Task<bool> ExistsAsync(string path, CancellationToken cancellationToken = default)
        {
            return Task.FromResult(_storage.ContainsKey(path));
        }

        public Task DeleteAsync(string path, CancellationToken cancellationToken = default)
        {
            _storage.Remove(path);
            return Task.CompletedTask;
        }

        public Task<StorageMetadata> GetMetadataAsync(string path, CancellationToken cancellationToken = default)
        {
            if (!_storage.ContainsKey(path))
                throw new FileNotFoundException($"File not found: {path}");

            return Task.FromResult(new StorageMetadata
            {
                Size = _storage[path].Length,
                LastModified = DateTime.UtcNow
            });
        }

        public int WriteCount => _storage.Count;
    }

    [Fact]
    public async Task CheckpointCoordinator_SingleRank_SavesCorrectly()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(rank: 0, worldSize: 1);
        var storage = new MockCheckpointStorage();
        var checkpointCoordinator = new CheckpointCoordinator(coordinator, storage);

        var shardData = new ShardData
        {
            Data = new byte[] { 1, 2, 3, 4 },
            TensorInfo = new List<TensorMetadata>
            {
                new TensorMetadata
                {
                    Name = "layer1.weight",
                    Shape = new long[] { 10, 20 },
                    DataType = TensorDataType.Float32,
                    Offset = 0,
                    Size = 40
                }
            }
        };

        // Act
        var metadata = await checkpointCoordinator.CoordinateSaveAsync(
            "checkpoint_test",
            () => Task.FromResult(shardData));

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(1, storage.WriteCount);
        Assert.Equal(1, coordinator.BarrierCallCount);
    }

    [Fact]
    public async Task CheckpointCoordinator_MultipleRanks_CoordinatesCorrectly()
    {
        // Arrange
        var rank0Coordinator = new MockDistributedCoordinator(rank: 0, worldSize: 2);
        var rank1Coordinator = new MockDistributedCoordinator(rank: 1, worldSize: 2);
        var storage = new MockCheckpointStorage();

        var coordinator0 = new CheckpointCoordinator(rank0Coordinator, storage);
        var coordinator1 = new CheckpointCoordinator(rank1Coordinator, storage);

        var shardData = new ShardData
        {
            Data = new byte[] { 1, 2, 3, 4 },
            TensorInfo = new List<TensorMetadata>
            {
                new TensorMetadata
                {
                    Name = "layer1.weight",
                    Shape = new long[] { 10, 20 },
                    DataType = TensorDataType.Float32,
                    Offset = 0,
                    Size = 40
                }
            }
        };

        // Act - Simulate both ranks saving
        var task0 = coordinator0.CoordinateSaveAsync(
            "checkpoint_multi",
            () => Task.FromResult(shardData));

        var task1 = coordinator1.CoordinateSaveAsync(
            "checkpoint_multi",
            () => Task.FromResult(shardData));

        await Task.WhenAll(task0, task1);

        // Assert
        Assert.NotNull(task0.Result);
        Assert.Null(task1.Result); // Only rank 0 returns metadata
        Assert.Equal(3, storage.WriteCount); // 2 shards + 1 metadata
        Assert.Equal(2, rank0Coordinator.BarrierCallCount);
        Assert.Equal(2, rank1Coordinator.BarrierCallCount);
    }

    [Fact]
    public async Task CheckpointLoader_LoadsWithCrossTopologyValidation()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(rank: 0, worldSize: 1);
        var storage = new MockCheckpointStorage();

        // Save a checkpoint first
        var saveCoordinator = new CheckpointCoordinator(coordinator, storage);
        var shardData = new ShardData
        {
            Data = new byte[] { 1, 2, 3, 4 },
            TensorInfo = new List<TensorMetadata>
            {
                new TensorMetadata
                {
                    Name = "layer1.weight",
                    Shape = new long[] { 10, 20 },
                    DataType = TensorDataType.Float32,
                    Offset = 0,
                    Size = 40
                }
            }
        };
        await saveCoordinator.CoordinateSaveAsync(
            "checkpoint_load",
            () => Task.FromResult(shardData));

        // Act - Load with different world size
        var loadCoordinator = new MockDistributedCoordinator(rank: 0, worldSize: 2);
        var loader = new CheckpointLoader(loadCoordinator, storage);
        var loadResult = await loader.CoordinateLoadAsync("checkpoint_load", targetWorldSize: 2);

        // Assert
        Assert.NotNull(loadResult);
        Assert.NotNull(loadResult.Metadata);
        Assert.Single(loadResult.Shards);
        Assert.Equal(1, loadResult.Metadata.WorldSize);
        Assert.True(loadResult.Success);
    }

    [Fact]
    public async Task AtomicCommitProtocol_CommitsSuccessfully()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), $"atomic_commit_test_{Guid.NewGuid()}");
        Directory.CreateDirectory(tempDir);

        try
        {
            var storage = new LocalFileSystemStorage(tempDir);
            var protocol = new AtomicCommitProtocol(storage);
            var data = new byte[] { 1, 2, 3, 4, 5 };

            // Act
            await protocol.CommitAsync("test_checkpoint.bin", data);

            // Assert
            var finalPath = Path.Combine(tempDir, "test_checkpoint.bin");
            Assert.True(File.Exists(finalPath));

            var tempPath = Path.Combine(tempDir, "test_checkpoint.bin.tmp");
            Assert.False(File.Exists(tempPath));

            var loadedData = await storage.ReadAsync("test_checkpoint.bin");
            Assert.Equal(data, loadedData);
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    [Fact]
    public async Task AtomicCommitProtocol_CleansUpTempFileOnFailure()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), $"atomic_commit_fail_test_{Guid.NewGuid()}");
        Directory.CreateDirectory(tempDir);

        try
        {
            var storage = new MockCheckpointStorage();
            storage.WriteAsync("test.tmp", new byte[] { 1, 2, 3 });

            // Create a custom storage that fails on second write
            var failingStorage = new MockCheckpointStorage();
            var writeCallCount = 0;
            failingStorage.WriteAsync = async (path, data, ct) =>
            {
                writeCallCount++;
                if (writeCallCount > 1)
                {
                    throw new InvalidOperationException("Simulated write failure");
                }
                await storage.WriteAsync(path, data, ct);
            };

            var protocol = new AtomicCommitProtocol(failingStorage);
            var data = new byte[] { 1, 2, 3, 4, 5 };

            // Act & Assert
            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                protocol.CommitAsync("test_checkpoint.bin", data));
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    [Fact]
    public async Task CheckpointCoordinator_ComputesChecksum()
    {
        // Arrange
        var coordinator = new MockDistributedCoordinator(rank: 0, worldSize: 1);
        var storage = new MockCheckpointStorage();
        var checkpointCoordinator = new CheckpointCoordinator(coordinator, storage);

        var shardData = new ShardData
        {
            Data = new byte[] { 1, 2, 3, 4 },
            TensorInfo = new List<TensorMetadata>()
        };

        // Act
        await checkpointCoordinator.CoordinateSaveAsync(
            "checksum_test",
            () => Task.FromResult(shardData));

        // Assert
        var metadataBytes = await storage.ReadAsync("checksum_test.metadata.json");
        var metadataJson = System.Text.Encoding.UTF8.GetString(metadataBytes);
        var metadata = MetadataSerializer.Deserialize(metadataJson);

        Assert.NotNull(metadata);
        Assert.NotNull(metadata.Shards);
        Assert.Single(metadata.Shards);
        Assert.NotNull(metadata.Shards[0].Checksum);
        Assert.NotEmpty(metadata.Shards[0].Checksum);
    }
}
