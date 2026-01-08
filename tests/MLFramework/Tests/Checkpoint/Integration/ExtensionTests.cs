namespace MachineLearning.Checkpointing.Tests;

using MachineLearning.Distributed.Checkpointing;
using Xunit;

/// <summary>
/// Tests for DistributedCheckpointExtension
/// </summary>
public class DistributedCheckpointExtensionTests : IDisposable
{
    private readonly string _checkpointDir;
    private readonly ElasticCheckpointManager _checkpointManager;

    public DistributedCheckpointExtensionTests()
    {
        // Create a temporary checkpoint directory
        _checkpointDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        _checkpointManager = new ElasticCheckpointManager(_checkpointDir);
    }

    [Fact]
    public async Task SaveDistributedAsync_FSDPModel_SavesCheckpoint()
    {
        // Arrange
        var stateDict = new StateDict
        {
            ["layer1.weight"] = new MockTensor(new long[] { 128, 256 }, TensorDataType.Float32, 128 * 256 * 4),
            ["layer1.bias"] = new MockTensor(new long[] { 128 }, TensorDataType.Float32, 128 * 4)
        };
        var model = new MockDistributedModel(DistributedStrategy.FullyShardedDataParallel, 4, 1, stateDict);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var optimizer = new MockStateful();
        var checkpoint = new DistributedCheckpoint(coordinator, _checkpointManager);
        var options = new SaveOptions { CheckpointPrefix = "fsdp_test" };

        // Act
        var checkpointId = await checkpoint.SaveDistributedAsync(model, optimizer, options);

        // Assert
        Assert.NotNull(checkpointId);
        Assert.NotEmpty(checkpointId);
        Assert.StartsWith("fsdp_test", checkpointId);

        // Verify checkpoint was saved
        var savedCheckpoint = await _checkpointManager.LoadCheckpointAsync(checkpointId);
        Assert.NotNull(savedCheckpoint);
        Assert.Equal(4, savedCheckpoint.WorkerCount);
        Assert.Equal("fsdp", savedCheckpoint.Metadata["strategy"]);
    }

    [Fact]
    public async Task SaveDistributedAsync_DDPModel_SavesCheckpoint()
    {
        // Arrange
        var stateDict = new StateDict
        {
            ["layer1.weight"] = new MockTensor(new long[] { 128, 256 }, TensorDataType.Float32, 128 * 256 * 4),
            ["layer1.bias"] = new MockTensor(new long[] { 128 }, TensorDataType.Float32, 128 * 4)
        };
        var model = new MockDistributedModel(DistributedStrategy.DataParallel, 4, 1, stateDict);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var optimizer = new MockStateful();
        var checkpoint = new DistributedCheckpoint(coordinator, _checkpointManager);
        var options = new SaveOptions { CheckpointPrefix = "ddp_test" };

        // Act
        var checkpointId = await checkpoint.SaveDistributedAsync(model, optimizer, options);

        // Assert
        Assert.NotNull(checkpointId);
        Assert.NotEmpty(checkpointId);
        Assert.StartsWith("ddp_test", checkpointId);

        // Verify checkpoint was saved
        var savedCheckpoint = await _checkpointManager.LoadCheckpointAsync(checkpointId);
        Assert.NotNull(savedCheckpoint);
        Assert.Equal(4, savedCheckpoint.WorkerCount);
        Assert.Equal("ddp", savedCheckpoint.Metadata["strategy"]);
    }

    [Fact]
    public async Task SaveDistributedAsync_TPModel_SavesCheckpoint()
    {
        // Arrange
        var stateDict = new StateDict
        {
            ["layer1.weight"] = new MockTensor(new long[] { 128, 64 }, TensorDataType.Float32, 128 * 64 * 4),
            ["layer1.bias"] = new MockTensor(new long[] { 128 }, TensorDataType.Float32, 128 * 4)
        };
        var model = new MockDistributedModel(DistributedStrategy.TensorParallel, 4, 1, stateDict);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var optimizer = new MockStateful();
        var checkpoint = new DistributedCheckpoint(coordinator, _checkpointManager);
        var options = new SaveOptions { CheckpointPrefix = "tp_test" };

        // Act
        var checkpointId = await checkpoint.SaveDistributedAsync(model, optimizer, options);

        // Assert
        Assert.NotNull(checkpointId);
        Assert.NotEmpty(checkpointId);
        Assert.StartsWith("tp_test", checkpointId);

        // Verify checkpoint was saved
        var savedCheckpoint = await _checkpointManager.LoadCheckpointAsync(checkpointId);
        Assert.NotNull(savedCheckpoint);
        Assert.Equal(4, savedCheckpoint.WorkerCount);
        Assert.Equal("tensor_parallel", savedCheckpoint.Metadata["strategy"]);
    }

    [Fact]
    public async Task SaveDistributedAsync_UnsupportedStrategy_ThrowsNotSupportedException()
    {
        // Arrange
        var model = new MockDistributedModel(DistributedStrategy.PipelineParallel, 4, 1);
        var coordinator = new MockDistributedCoordinator(4, 1);
        var optimizer = new MockStateful();
        var checkpoint = new DistributedCheckpoint(coordinator, _checkpointManager);
        var options = new SaveOptions { CheckpointPrefix = "test" };

        // Act & Assert
        await Assert.ThrowsAsync<NotSupportedException>(() =>
            checkpoint.SaveDistributedAsync(model, optimizer, options));
    }

    public void Dispose()
    {
        // Clean up the temporary checkpoint directory
        if (Directory.Exists(_checkpointDir))
        {
            try
            {
                Directory.Delete(_checkpointDir, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }
}
