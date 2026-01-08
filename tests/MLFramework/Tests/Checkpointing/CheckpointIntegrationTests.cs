namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Integration tests for DistributedCheckpoint
/// </summary>
public class DistributedCheckpointIntegrationTests
{
    private string _testDirectory = string.Empty;
    private ICheckpointStorage _storage = null!;
    private MockDistributedCoordinator _mockCoordinator = null!;

    private void SetUp()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"checkpoint_integration_test_{Guid.NewGuid():N}");
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
    public async Task SaveAndLoad_Roundtrip_Success()
    {
        // Arrange
        SetUp();
        try
        {
            var model = new MockModel();
            var optimizer = new MockOptimizer();
            var checkpoint = new DistributedCheckpoint(_mockCoordinator, _storage);

            // Act
            var checkpointPath = await checkpoint.SaveAsync(model, optimizer);
            var loadResult = await checkpoint.LoadAsync(model, optimizer, new LoadOptions
            {
                CheckpointPrefix = checkpointPath
            });

            // Assert
            Assert.NotNull(loadResult.Metadata);
            Assert.True(loadResult.Success);
        }
        finally
        {
            TearDown();
        }
    }

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
                ["state"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 1024)
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
