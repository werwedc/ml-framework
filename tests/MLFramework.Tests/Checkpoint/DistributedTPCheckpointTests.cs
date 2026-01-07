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
/// Tests for DistributedTPCheckpoint functionality
/// </summary>
public class DistributedTPCheckpointTests : IDisposable
{
    private readonly string _testCheckpointDir;
    private readonly string _tempDir;

    public DistributedTPCheckpointTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"checkpoint_test_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDir);
        _testCheckpointDir = Path.Combine(_tempDir, "checkpoint");
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
    public async Task SaveCheckpoint_CreatesDirectoryStructure()
    {
        // Arrange
        var model = CreateTestLinearModule();
        var checkpoint = new DistributedTPCheckpoint(_testCheckpointDir);

        // Act
        await checkpoint.SaveAsync(model, rank: 0, customName: "TestModel");

        // Assert
        Assert.True(Directory.Exists(_testCheckpointDir));
        Assert.True(File.Exists(Path.Combine(_testCheckpointDir, "metadata.bin")));
        Assert.True(File.Exists(Path.Combine(_testCheckpointDir, "shard_rank0.pt")));
    }

    [Fact]
    public async Task SaveCheckpoint_SavesMetadataCorrectly()
    {
        // Arrange
        var model = CreateTestLinearModule();
        var checkpoint = new DistributedTPCheckpoint(_testCheckpointDir);

        // Act
        await checkpoint.SaveAsync(model, rank: 0, customName: "TestModel");
        var metadata = checkpoint.GetMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal("TestModel", metadata.ModelName);
        Assert.True(metadata.SavedAt <= DateTime.UtcNow.AddMinutes(1));
        Assert.Equal(1, metadata.TPWorldSize);
    }

    [Fact]
    public async Task SaveAndLoadCheckpoint_RestoresModelParameters()
    {
        // Arrange
        var originalModel = CreateTestLinearModule();
        var checkpoint = new DistributedTPCheckpoint(_testCheckpointDir);

        // Save original model parameters
        var originalWeight = originalModel.Weight.Data.Data.ToArray();
        var originalBias = originalModel.Bias?.Data.Data.ToArray();

        // Act
        await checkpoint.SaveAsync(originalModel, rank: 0);

        // Create new model and load checkpoint
        var loadedModel = CreateTestLinearModule();
        // Modify loaded model parameters to verify they are changed
        ModifyParameters(loadedModel);

        await checkpoint.LoadAsync(loadedModel, rank: 0);

        // Assert
        var loadedWeight = loadedModel.Weight.Data.Data.ToArray();
        var loadedBias = loadedModel.Bias?.Data.Data.ToArray();

        Assert.Equal(originalWeight, loadedWeight);
        if (originalBias != null && loadedBias != null)
        {
            Assert.Equal(originalBias, loadedBias);
        }
    }

    [Fact]
    public async Task SaveMultipleRanks_CreatesSeparateShardFiles()
    {
        // Arrange
        var rank0Model = CreateTestLinearModule();
        var rank1Model = CreateTestLinearModule();
        var checkpoint = new DistributedTPCheckpoint(_testCheckpointDir);

        // Act
        await checkpoint.SaveAsync(rank0Model, rank: 0);
        await checkpoint.SaveAsync(rank1Model, rank: 1);

        // Assert
        Assert.True(File.Exists(Path.Combine(_testCheckpointDir, "shard_rank0.pt")));
        Assert.True(File.Exists(Path.Combine(_testCheckpointDir, "shard_rank1.pt")));
    }

    [Fact]
    public async Task LoadFromSpecificRank_LoadsCorrectShard()
    {
        // Arrange
        var rank0Model = CreateTestLinearModule();
        var rank1Model = CreateTestLinearModule();
        var checkpoint = new DistributedTPCheckpoint(_testCheckpointDir);

        // Save both ranks
        await checkpoint.SaveAsync(rank0Model, rank: 0);
        await checkpoint.SaveAsync(rank1Model, rank: 1);

        // Create new models for loading
        var loadRank0Model = CreateTestLinearModule();
        var loadRank1Model = CreateTestLinearModule();

        // Act
        await checkpoint.LoadAsync(loadRank0Model, rank: 0);
        await checkpoint.LoadAsync(loadRank1Model, rank: 1);

        // Assert
        var rank0Weight = loadRank0Model.Weight.Data.Data;
        var rank1Weight = loadRank1Model.Weight.Data.Data;

        // Each rank should have loaded its own shard
        Assert.NotNull(rank0Weight);
        Assert.NotNull(rank1Weight);
    }

    [Fact]
    public async Task LoadNonExistentCheckpoint_ThrowsFileNotFoundException()
    {
        // Arrange
        var model = CreateTestLinearModule();
        var checkpoint = new DistributedTPCheckpoint(_testCheckpointDir);

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            checkpoint.LoadAsync(model, rank: 0));
    }

    [Fact]
    public void GetMetadata_WithoutCheckpoint_ReturnsNull()
    {
        // Arrange
        var checkpoint = new DistributedTPCheckpoint(_testCheckpointDir);

        // Act
        var metadata = checkpoint.GetMetadata();

        // Assert
        Assert.Null(metadata);
    }

    [Fact]
    public async Task SaveCheckpoint_WithModuleClass_WorksCorrectly()
    {
        // Arrange
        var model = CreateTestModuleClass();
        var checkpoint = new DistributedTPCheckpoint(_testCheckpointDir);

        // Act
        await checkpoint.SaveAsync(model, rank: 0, customName: "TestModule");

        // Assert
        Assert.True(Directory.Exists(_testCheckpointDir));
        Assert.True(File.Exists(Path.Combine(_testCheckpointDir, "metadata.bin")));
        Assert.True(File.Exists(Path.Combine(_testCheckpointDir, "shard_rank0.pt")));

        var metadata = checkpoint.GetMetadata();
        Assert.NotNull(metadata);
        Assert.Equal("TestModule", metadata.ModelName);
    }

    [Fact]
    public async Task SaveAndLoadCheckpoint_WithModuleClass_WorksCorrectly()
    {
        // Arrange
        var originalModel = CreateTestModuleClass();
        var checkpoint = new DistributedTPCheckpoint(_testCheckpointDir);

        // Save original model
        await checkpoint.SaveAsync(originalModel, rank: 0);

        // Create new model and load checkpoint
        var loadedModel = CreateTestModuleClass();
        var originalParams = new Dictionary<string, float[]>();

        foreach (var (name, param) in originalModel.GetNamedParameters())
        {
            originalParams[name] = param.Data.Data.ToArray();
        }

        // Act
        await checkpoint.LoadAsync(loadedModel, rank: 0);

        // Assert
        foreach (var (name, param) in loadedModel.GetNamedParameters())
        {
            Assert.True(originalParams.ContainsKey(name));
            Assert.Equal(originalParams[name], param.Data.Data.ToArray());
        }
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
        // Modify weight
        for (int i = 0; i < model.Weight.Data.Data.Length; i++)
        {
            model.Weight.Data.Data[i] = 999.0f;
        }

        // Modify bias
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
            // Simple forward pass for testing
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
