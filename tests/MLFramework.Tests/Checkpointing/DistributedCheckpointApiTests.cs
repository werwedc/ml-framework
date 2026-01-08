namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for DistributedCheckpoint API
/// </summary>
public class DistributedCheckpointApiTests
{
    private string _testDirectory = string.Empty;
    private ICheckpointStorage _storage = null!;
    private MockDistributedCoordinator _mockCoordinator = null!;

    private void SetUp()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"checkpoint_api_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDirectory);

        _storage = new LocalFileSystemStorage(_testDirectory);
        _mockCoordinator = new MockDistributedCoordinator(1, 0);
    }

    private void TearDown()
    {
        if (!string.IsNullOrEmpty(_testDirectory) && Directory.Exists(_testDirectory))
        {
            try
            {
                Directory.Delete(_testDirectory, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    [Fact]
    public async Task SaveAsync_WithOptions_Success()
    {
        // Arrange
        SetUp();
        try
        {
            var model = new MockModel();
            var optimizer = new MockOptimizer();
            var checkpoint = DistributedCheckpointFactory.Create(_mockCoordinator);

            // Act
            var checkpointPath = await checkpoint.SaveAsync(model, optimizer, new SaveOptions
            {
                CheckpointPrefix = "model_epoch_100",
                Format = CheckpointFormat.MultiShard,
                CompressionLevel = CheckpointCompressionLevel.None,
                IncludeOptimizer = true,
                IncludeRngState = true
            });

            // Assert
            Assert.NotNull(checkpointPath);
            Assert.True(checkpointPath.Contains("model_epoch_100") || checkpointPath.Contains("checkpoint_"));
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task SaveAsync_AutoGeneratePrefix_Success()
    {
        // Arrange
        SetUp();
        try
        {
            var model = new MockModel();
            var optimizer = new MockOptimizer();
            var checkpoint = DistributedCheckpointFactory.Create(_mockCoordinator);

            // Act
            var checkpointPath = await checkpoint.SaveAsync(model, optimizer, new SaveOptions
            {
                CheckpointPrefix = null
            });

            // Assert
            Assert.NotNull(checkpointPath);
            Assert.StartsWith("checkpoint_", checkpointPath);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task LoadAsync_WithOptions_Success()
    {
        // Arrange
        SetUp();
        try
        {
            var model = new MockModel();
            var optimizer = new MockOptimizer();
            var checkpoint = DistributedCheckpointFactory.Create(_mockCoordinator);

            // Save checkpoint
            var checkpointPath = await checkpoint.SaveAsync(model, optimizer, new SaveOptions
            {
                CheckpointPrefix = "test_checkpoint"
            });

            // Act
            var loadResult = await checkpoint.LoadAsync(model, optimizer, new LoadOptions
            {
                CheckpointPrefix = checkpointPath,
                LoadOptimizer = true,
                LoadRngState = true,
                SkipValidation = false,
                StrictMode = true
            });

            // Assert
            Assert.NotNull(loadResult);
            Assert.True(loadResult.Success);
            Assert.NotNull(loadResult.Metadata);
            Assert.Equal(1, loadResult.SourceWorldSize);
            Assert.Equal(1, loadResult.TargetWorldSize);
            Assert.False(loadResult.WasResharded);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task LoadAsync_WithValidation_Success()
    {
        // Arrange
        SetUp();
        try
        {
            var model = new MockModel();
            var optimizer = new MockOptimizer();
            var checkpoint = DistributedCheckpointFactory.Create(_mockCoordinator);

            // Save checkpoint
            var checkpointPath = await checkpoint.SaveAsync(model, optimizer);

            // Act
            var loadResult = await checkpoint.LoadAsync(model, optimizer, new LoadOptions
            {
                CheckpointPrefix = checkpointPath,
                SkipValidation = false
            });

            // Assert
            Assert.True(loadResult.Success);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task LoadAsync_WithoutValidation_Success()
    {
        // Arrange
        SetUp();
        try
        {
            var model = new MockModel();
            var optimizer = new MockOptimizer();
            var checkpoint = DistributedCheckpointFactory.Create(_mockCoordinator);

            // Save checkpoint
            var checkpointPath = await checkpoint.SaveAsync(model, optimizer);

            // Act
            var loadResult = await checkpoint.LoadAsync(model, optimizer, new LoadOptions
            {
                CheckpointPrefix = checkpointPath,
                SkipValidation = true
            });

            // Assert
            Assert.True(loadResult.Success);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task SaveModelOnlyAsync_Success()
    {
        // Arrange
        SetUp();
        try
        {
            var model = new MockModel();
            var checkpoint = DistributedCheckpointFactory.Create(_mockCoordinator);

            // Act
            var checkpointPath = await checkpoint.SaveModelOnlyAsync(model, new SaveOptions
            {
                CheckpointPrefix = "model_only_checkpoint"
            });

            // Assert
            Assert.NotNull(checkpointPath);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task ListCheckpointsAsync_ReturnsEmptyList()
    {
        // Arrange
        SetUp();
        try
        {
            var checkpoint = DistributedCheckpointFactory.Create(_mockCoordinator);

            // Act
            var checkpoints = await checkpoint.ListCheckpointsAsync();

            // Assert
            Assert.NotNull(checkpoints);
            Assert.Empty(checkpoints);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task DeleteCheckpointAsync_Success()
    {
        // Arrange
        SetUp();
        try
        {
            var model = new MockModel();
            var optimizer = new MockOptimizer();
            var checkpoint = DistributedCheckpointFactory.Create(_mockCoordinator);

            // Save checkpoint
            var checkpointPath = await checkpoint.SaveAsync(model, optimizer);

            // Act
            await checkpoint.DeleteCheckpointAsync(checkpointPath);

            // Assert - no exception thrown
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task SaveAndLoad_RoundtripWithCustomMetadata_Success()
    {
        // Arrange
        SetUp();
        try
        {
            var model = new MockModel();
            var optimizer = new MockOptimizer();
            var checkpoint = DistributedCheckpointFactory.Create(_mockCoordinator);

            // Act
            var checkpointPath = await checkpoint.SaveAsync(model, optimizer, new SaveOptions
            {
                CheckpointPrefix = "custom_metadata_test",
                CustomMetadata = new Dictionary<string, object>
                {
                    ["epoch"] = 100,
                    ["loss"] = 0.1234f,
                    ["model_name"] = "test_model"
                }
            });

            var loadResult = await checkpoint.LoadAsync(model, optimizer, new LoadOptions
            {
                CheckpointPrefix = checkpointPath
            });

            // Assert
            Assert.True(loadResult.Success);
            Assert.NotNull(loadResult.Metadata);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public void DistributedCheckpointFactory_Create_ReturnsValidCheckpoint()
    {
        // Arrange
        SetUp();
        try
        {
            // Act
            var checkpoint = DistributedCheckpointFactory.Create(_mockCoordinator);

            // Assert
            Assert.NotNull(checkpoint);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public void DistributedCheckpointFactory_CreateWithOptions_ReturnsValidCheckpoint()
    {
        // Arrange
        SetUp();
        try
        {
            var options = new CheckpointOptions
            {
                RetryPolicy = new RetryPolicy(5, TimeSpan.FromSeconds(2)),
                DefaultTimeout = TimeSpan.FromMinutes(5)
            };

            // Act
            var checkpoint = DistributedCheckpointFactory.Create(_mockCoordinator, options);

            // Assert
            Assert.NotNull(checkpoint);
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public async Task LoadAsync_InvalidPrefix_ThrowsException()
    {
        // Arrange
        SetUp();
        try
        {
            var model = new MockModel();
            var optimizer = new MockOptimizer();
            var checkpoint = DistributedCheckpointFactory.Create(_mockCoordinator);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(() =>
                checkpoint.LoadAsync(model, optimizer, new LoadOptions
                {
                    CheckpointPrefix = null
                }));

            await Assert.ThrowsAsync<ArgumentException>(() =>
                checkpoint.LoadAsync(model, optimizer, new LoadOptions
                {
                    CheckpointPrefix = ""
                }));
        }
        finally
        {
            TearDown();
        }
    }

    [Fact]
    public void SaveOptions_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var options = new SaveOptions();

        // Assert
        Assert.Equal(CheckpointFormat.MultiShard, options.Format);
        Assert.Equal(CheckpointCompressionLevel.None, options.CompressionLevel);
        Assert.True(options.IncludeOptimizer);
        Assert.True(options.IncludeRngState);
        Assert.NotNull(options.CustomMetadata);
        Assert.Empty(options.CustomMetadata);
    }

    [Fact]
    public void LoadOptions_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var options = new LoadOptions();

        // Assert
        Assert.True(options.LoadOptimizer);
        Assert.True(options.LoadRngState);
        Assert.Equal("parallel", options.ReshardingStrategy);
        Assert.False(options.SkipValidation);
        Assert.True(options.StrictMode);
    }

    [Fact]
    public void RetryPolicy_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var policy = new RetryPolicy();

        // Assert
        Assert.Equal(3, policy.MaxRetries);
        Assert.Equal(TimeSpan.FromSeconds(1), policy.RetryDelay);
        Assert.False(policy.UseExponentialBackoff);
        Assert.Equal(TimeSpan.FromSeconds(30), policy.MaxBackoffDelay);
    }

    [Fact]
    public void CheckpointOptions_DefaultValues_AreCorrect()
    {
        // Arrange & Act
        var options = new CheckpointOptions();

        // Assert
        Assert.NotNull(options.Storage);
        Assert.NotNull(options.RetryPolicy);
        Assert.NotNull(options.IntegrityCheckers);
        Assert.NotNull(options.CompatibilityCheckers);
        Assert.Equal(TimeSpan.FromMinutes(10), options.DefaultTimeout);
    }

    // Helper classes
    private class MockModel : IStateful
    {
        private readonly StateDict _stateDict;

        public MockModel()
        {
            _stateDict = new StateDict
            {
                ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 1024)
            };
        }

        public StateDict GetStateDict()
        {
            return _stateDict;
        }

        public void LoadStateDict(StateDict state)
        {
            // Load state
            foreach (var (key, value) in state)
            {
                _stateDict[key] = value;
            }
        }
    }

    private class MockOptimizer : IStateful
    {
        private readonly StateDict _stateDict;

        public MockOptimizer()
        {
            _stateDict = new StateDict
            {
                ["momentum"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 1024)
            };
        }

        public StateDict GetStateDict()
        {
            return _stateDict;
        }

        public void LoadStateDict(StateDict state)
        {
            // Load state
            foreach (var (key, value) in state)
            {
                _stateDict[key] = value;
            }
        }
    }
}
