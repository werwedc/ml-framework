using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Checkpoint;
using MLFramework.Modules;
using Xunit;

namespace MLFramework.Tests.Checkpoint;

/// <summary>
/// Tests for TPCheckpointManager functionality
/// </summary>
public class TPCheckpointManagerTests : IDisposable
{
    private readonly string _tempDir;

    public TPCheckpointManagerTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"manager_test_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            try
            {
                Directory.Delete(_tempDir, true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    [Fact]
    public async Task SaveDistributedAsync_CreatesCheckpoint()
    {
        // Arrange
        var model = CreateTestLinearModule();
        var checkpointDir = Path.Combine(_tempDir, "distributed_checkpoint");

        // Act
        await TPCheckpointManager.SaveDistributedAsync(model, checkpointDir, "TestModel");

        // Assert
        Assert.True(TPCheckpointManager.CheckpointExists(checkpointDir, isDistributed: true));
        Assert.True(File.Exists(Path.Combine(checkpointDir, "metadata.bin")));
    }

    [Fact]
    public async Task LoadDistributedAsync_RestoresModel()
    {
        // Arrange
        var originalModel = CreateTestLinearModule();
        var checkpointDir = Path.Combine(_tempDir, "distributed_checkpoint");

        // Save original model
        await TPCheckpointManager.SaveDistributedAsync(originalModel, checkpointDir, "TestModel");

        // Create new model
        var loadedModel = CreateTestLinearModule();
        ModifyParameters(loadedModel);

        // Act
        await TPCheckpointManager.LoadDistributedAsync(loadedModel, checkpointDir);

        // Assert
        Assert.Equal(originalModel.Weight.Data.Data, loadedModel.Weight.Data.Data);
    }

    [Fact]
    public async Task SaveCentralizedAsync_CreatesSingleFile()
    {
        // Arrange
        var model = CreateTestLinearModule();
        var checkpointFile = Path.Combine(_tempDir, "centralized_checkpoint.pt");

        // Act
        await TPCheckpointManager.SaveCentralizedAsync(model, checkpointFile, "TestModel");

        // Assert
        Assert.True(TPCheckpointManager.CheckpointExists(checkpointFile, isDistributed: false));
        Assert.True(File.Exists(checkpointFile));
    }

    [Fact]
    public async Task LoadCentralizedAsync_RestoresModel()
    {
        // Arrange
        var originalModel = CreateTestLinearModule();
        var checkpointFile = Path.Combine(_tempDir, "centralized_checkpoint.pt");

        // Save original model
        await TPCheckpointManager.SaveCentralizedAsync(originalModel, checkpointFile, "TestModel");

        // Create new model
        var loadedModel = CreateTestLinearModule();
        ModifyParameters(loadedModel);

        // Act
        await TPCheckpointManager.LoadCentralizedAsync(loadedModel, checkpointFile);

        // Assert
        Assert.Equal(originalModel.Weight.Data.Data, loadedModel.Weight.Data.Data);
    }

    [Fact]
    public void CheckpointExists_DistributedCheckpoint_ReturnsTrue()
    {
        // Arrange
        var checkpointDir = Path.Combine(_tempDir, "checkpoint");
        Directory.CreateDirectory(checkpointDir);
        File.WriteAllText(Path.Combine(checkpointDir, "metadata.bin"), "dummy");

        // Act
        var exists = TPCheckpointManager.CheckpointExists(checkpointDir, isDistributed: true);

        // Assert
        Assert.True(exists);
    }

    [Fact]
    public void CheckpointExists_NonExistentCheckpoint_ReturnsFalse()
    {
        // Arrange
        var checkpointDir = Path.Combine(_tempDir, "nonexistent");

        // Act
        var exists = TPCheckpointManager.CheckpointExists(checkpointDir, isDistributed: true);

        // Assert
        Assert.False(exists);
    }

    [Fact]
    public void CheckpointExists_CentralizedCheckpoint_ReturnsTrue()
    {
        // Arrange
        var checkpointFile = Path.Combine(_tempDir, "checkpoint.pt");
        File.WriteAllText(checkpointFile, "dummy");

        // Act
        var exists = TPCheckpointManager.CheckpointExists(checkpointFile, isDistributed: false);

        // Assert
        Assert.True(exists);
    }

    [Fact]
    public async Task GetMetadata_ReturnsCorrectMetadata()
    {
        // Arrange
        var model = CreateTestLinearModule();
        var checkpointDir = Path.Combine(_tempDir, "checkpoint");

        await TPCheckpointManager.SaveDistributedAsync(model, checkpointDir, "MyModel");

        // Act
        var metadata = TPCheckpointManager.GetMetadata(checkpointDir);

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal("MyModel", metadata.ModelName);
    }

    [Fact]
    public void GetMetadata_NonExistentCheckpoint_ReturnsNull()
    {
        // Arrange
        var checkpointDir = Path.Combine(_tempDir, "nonexistent");

        // Act
        var metadata = TPCheckpointManager.GetMetadata(checkpointDir);

        // Assert
        Assert.Null(metadata);
    }

    [Fact]
    public void ListCheckpoints_ReturnsCorrectList()
    {
        // Arrange
        var checkpointsBaseDir = Path.Combine(_tempDir, "checkpoints");
        Directory.CreateDirectory(checkpointsBaseDir);

        var ckpt1 = Path.Combine(checkpointsBaseDir, "ckpt1");
        var ckpt2 = Path.Combine(checkpointsBaseDir, "ckpt2");
        var nonCkpt = Path.Combine(checkpointsBaseDir, "non_ckpt");

        Directory.CreateDirectory(ckpt1);
        Directory.CreateDirectory(ckpt2);
        Directory.CreateDirectory(nonCkpt);

        // Only create metadata files for checkpoints
        File.WriteAllText(Path.Combine(ckpt1, "metadata.bin"), "dummy");
        File.WriteAllText(Path.Combine(ckpt2, "metadata.bin"), "dummy");
        // Don't create metadata for non_ckpt

        // Act
        var checkpoints = TPCheckpointManager.ListCheckpoints(checkpointsBaseDir);

        // Assert
        Assert.Equal(2, checkpoints.Count);
        Assert.Contains("ckpt1", checkpoints);
        Assert.Contains("ckpt2", checkpoints);
        Assert.DoesNotContain("non_ckpt", checkpoints);
    }

    [Fact]
    public void ListCheckpoints_NonExistentDirectory_ReturnsEmptyList()
    {
        // Arrange
        var checkpointDir = Path.Combine(_tempDir, "nonexistent");

        // Act
        var checkpoints = TPCheckpointManager.ListCheckpoints(checkpointDir);

        // Assert
        Assert.Empty(checkpoints);
    }

    [Fact]
    public async Task SaveDistributedAsync_WithModuleClass_WorksCorrectly()
    {
        // Arrange
        var model = CreateTestModuleClass();
        var checkpointDir = Path.Combine(_tempDir, "module_checkpoint");

        // Act
        await TPCheckpointManager.SaveDistributedAsync(model, checkpointDir, "MyModule");

        // Assert
        Assert.True(TPCheckpointManager.CheckpointExists(checkpointDir, isDistributed: true));
        var metadata = TPCheckpointManager.GetMetadata(checkpointDir);
        Assert.Equal("MyModule", metadata.ModelName);
    }

    [Fact]
    public async Task SaveAndLoadModuleClass_WorksCorrectly()
    {
        // Arrange
        var originalModel = CreateTestModuleClass();
        var checkpointDir = Path.Combine(_tempDir, "module_checkpoint");

        await TPCheckpointManager.SaveDistributedAsync(originalModel, checkpointDir, "MyModule");

        // Create new model
        var loadedModel = CreateTestModuleClass();
        var originalParams = new Dictionary<string, float[]>();

        foreach (var (name, param) in originalModel.GetNamedParameters())
        {
            originalParams[name] = param.Data.Data.ToArray();
        }

        // Act
        await TPCheckpointManager.LoadDistributedAsync(loadedModel, checkpointDir);

        // Assert
        foreach (var (name, param) in loadedModel.GetNamedParameters())
        {
            Assert.True(originalParams.ContainsKey(name));
            Assert.Equal(originalParams[name], param.Data.Data.ToArray());
        }
    }

    [Fact]
    public async Task MultipleCheckpoints_CanBeSavedAndLoaded()
    {
        // Arrange
        var model1 = CreateTestLinearModule();
        var model2 = CreateTestLinearModule();

        var checkpointDir1 = Path.Combine(_tempDir, "checkpoints", "ckpt1");
        var checkpointDir2 = Path.Combine(_tempDir, "checkpoints", "ckpt2");

        // Act
        await TPCheckpointManager.SaveDistributedAsync(model1, checkpointDir1, "Model1");
        await TPCheckpointManager.SaveDistributedAsync(model2, checkpointDir2, "Model2");

        // Assert
        var checkpointsBaseDir = Path.Combine(_tempDir, "checkpoints");
        var checkpoints = TPCheckpointManager.ListCheckpoints(checkpointsBaseDir);

        Assert.Equal(2, checkpoints.Count);
        Assert.Contains("ckpt1", checkpoints);
        Assert.Contains("ckpt2", checkpoints);
    }

    // Helper methods

    private Linear CreateTestLinearModule()
    {
        return new Linear(inFeatures: 10, outFeatures: 5, useBias: true);
    }

    private TestModule CreateTestModuleClass()
    {
        return new TestModule();
    }

    private void ModifyParameters(Linear model)
    {
        for (int i = 0; i < model.Weight.Data.Data.Length; i++)
        {
            model.Weight.Data.Data[i] = 999.0f;
        }

        if (model.Bias != null)
        {
            for (int i = 0; i < model.Bias.Data.Data.Length; i++)
            {
                model.Bias.Data.Data[i] = 888.0f;
            }
        }
    }

    /// <summary>
    /// Simple test module that extends Module (abstract class)
    /// </summary>
    private class TestModule : NN.Module
    {
        private readonly Parameter _param1;
        private readonly Parameter _param2;

        public TestModule() : base("TestModule")
        {
            _param1 = new Parameter(
                new float[] { 1, 2, 3, 4 },
                new[] { 2, 2 },
                "param1",
                requiresGrad: false,
                dtype: DataType.Float32
            );

            _param2 = new Parameter(
                new float[] { 5, 6, 7, 8 },
                new[] { 2, 2 },
                "param2",
                requiresGrad: false,
                dtype: DataType.Float32
            );
        }

        public override Tensor Forward(Tensor input)
        {
            return input;
        }

        public override IEnumerable<NN.Parameter> GetParameters()
        {
            yield return _param1;
            yield return _param2;
        }

        public override IEnumerable<(string Name, NN.Parameter Parameter)> GetNamedParameters()
        {
            yield return ("param1", _param1);
            yield return ("param2", _param2);
        }
    }
}
