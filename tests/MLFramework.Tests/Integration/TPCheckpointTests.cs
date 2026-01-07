using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.Tests.Integration;
using MLFramework.Tensor;
using System;
using System.IO;

namespace MLFramework.Tests.Integration
{
    [TestClass]
    public class TPCheckpointTests
    {
        private string GetTestCheckpointPath()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
            Directory.CreateDirectory(tempPath);
            return tempPath;
        }

        [TestMethod]
        public async System.Threading.Tasks.Task TPCheckpoint_SaveAndLoad_RestoresModelCorrectly()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
            var checkpointDir = GetTestCheckpointPath();

            var originalModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
            var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10);

            var originalOutput = originalModel.Forward(input);

            // Act: Save checkpoint
            await TPCheckpointManager.SaveDistributedAsync(originalModel, checkpointDir);

            // Create new model and load checkpoint
            var loadedModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5, context: tpContext);
            await TPCheckpointManager.LoadDistributedAsync(loadedModel, checkpointDir);

            var loadedOutput = loadedModel.Forward(input);

            // Assert: Outputs should match
            Assert.IsTrue(
                TPTestHelpers.TensorsApproxEqual(originalOutput, loadedOutput, tolerance: 1e-6),
                "Loaded model should produce same output as original");

            // Cleanup
            Directory.Delete(checkpointDir, recursive: true);
        }

        [TestMethod]
        public void TPCheckpoint_ListCheckpoints_ReturnsCorrectList()
        {
            // Arrange
            var rootDir = GetTestCheckpointPath();

            // Create multiple checkpoints
            var checkpoint1 = Path.Combine(rootDir, "checkpoint1");
            var checkpoint2 = Path.Combine(rootDir, "checkpoint2");

            using (var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0))
            {
                var model = TPTestHelpers.CreateSimpleTPMLP(10, 20, 5, context: tpContext);
                TPCheckpointManager.SaveDistributedAsync(model, checkpoint1).Wait();
                TPCheckpointManager.SaveDistributedAsync(model, checkpoint2).Wait();
            }

            // Act
            var checkpoints = TPCheckpointManager.ListCheckpoints(rootDir);

            // Assert
            Assert.AreEqual(2, checkpoints.Count);
            CollectionAssert.Contains(checkpoints, "checkpoint1");
            CollectionAssert.Contains(checkpoints, "checkpoint2");

            // Cleanup
            Directory.Delete(rootDir, recursive: true);
        }

        [TestMethod]
        public async System.Threading.Tasks.Task TPCheckpoint_Save_CreatesDirectory()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
            var checkpointDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
            var model = TPTestHelpers.CreateSimpleTPMLP(10, 20, 5, context: tpContext);

            // Act
            await TPCheckpointManager.SaveDistributedAsync(model, checkpointDir);

            // Assert
            Assert.IsTrue(Directory.Exists(checkpointDir), "Checkpoint directory should be created");

            // Cleanup
            Directory.Delete(checkpointDir, recursive: true);
        }

        [TestMethod]
        public void TPCheckpoint_ListCheckpoints_EmptyDirectory_ReturnsEmpty()
        {
            // Arrange
            var rootDir = GetTestCheckpointPath();

            // Act
            var checkpoints = TPCheckpointManager.ListCheckpoints(rootDir);

            // Assert
            Assert.AreEqual(0, checkpoints.Count);

            // Cleanup
            Directory.Delete(rootDir, recursive: true);
        }

        [TestMethod]
        public void TPCheckpoint_ListCheckpoints_NonExistentDirectory_ReturnsEmpty()
        {
            // Arrange
            var nonExistentDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());

            // Act
            var checkpoints = TPCheckpointManager.ListCheckpoints(nonExistentDir);

            // Assert
            Assert.AreEqual(0, checkpoints.Count);
        }

        [TestMethod]
        public async System.Threading.Tasks.Task TPCheckpoint_MultipleCheckpoints_AllSaved()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
            var rootDir = GetTestCheckpointPath();
            var model = TPTestHelpers.CreateSimpleTPMLP(10, 20, 5, context: tpContext);

            // Act: Save multiple checkpoints
            for (int i = 0; i < 5; i++)
            {
                var checkpointDir = Path.Combine(rootDir, $"checkpoint{i}");
                await TPCheckpointManager.SaveDistributedAsync(model, checkpointDir);
            }

            // Assert
            var checkpoints = TPCheckpointManager.ListCheckpoints(rootDir);
            Assert.AreEqual(5, checkpoints.Count);

            // Cleanup
            Directory.Delete(rootDir, recursive: true);
        }

        [TestMethod]
        public async System.Threading.Tasks.Task TPCheckpoint_LoadAfterSave_Works()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
            var checkpointDir = GetTestCheckpointPath();
            var model = TPTestHelpers.CreateSimpleTPMLP(10, 20, 5, context: tpContext);

            // Act
            await TPCheckpointManager.SaveDistributedAsync(model, checkpointDir);
            await TPCheckpointManager.LoadDistributedAsync(model, checkpointDir);

            // Assert: Should complete without errors
            Assert.IsTrue(true);

            // Cleanup
            Directory.Delete(checkpointDir, recursive: true);
        }

        [TestMethod]
        public async System.Threading.Tasks.Task TPCheckpoint_SaveDifferentModels_AllSaved()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
            var rootDir = GetTestCheckpointPath();

            // Act: Save different models
            var model1 = TPTestHelpers.CreateSimpleTPMLP(10, 20, 5, context: tpContext);
            var model2 = TPTestHelpers.CreateSimpleTPMLP(15, 30, 10, context: tpContext);
            var model3 = TPTestHelpers.CreateSimpleTPMLP(20, 40, 15, context: tpContext);

            await TPCheckpointManager.SaveDistributedAsync(model1, Path.Combine(rootDir, "model1"));
            await TPCheckpointManager.SaveDistributedAsync(model2, Path.Combine(rootDir, "model2"));
            await TPCheckpointManager.SaveDistributedAsync(model3, Path.Combine(rootDir, "model3"));

            // Assert
            var checkpoints = TPCheckpointManager.ListCheckpoints(rootDir);
            Assert.AreEqual(3, checkpoints.Count);

            // Cleanup
            Directory.Delete(rootDir, recursive: true);
        }

        [TestMethod]
        public async System.Threading.Tasks.Task TPCheckpoint_NestedPath_Works()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
            var rootDir = GetTestCheckpointPath();
            var nestedDir = Path.Combine(rootDir, "level1", "level2", "checkpoint");
            var model = TPTestHelpers.CreateSimpleTPMLP(10, 20, 5, context: tpContext);

            // Act
            await TPCheckpointManager.SaveDistributedAsync(model, nestedDir);

            // Assert
            Assert.IsTrue(Directory.Exists(nestedDir), "Nested directories should be created");

            // Cleanup
            Directory.Delete(rootDir, recursive: true);
        }

        [TestMethod]
        public async System.Threading.Tasks.Task TPCheckpoint_SaveSameLocation_Overwrites()
        {
            // Arrange
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
            var checkpointDir = GetTestCheckpointPath();

            // Act: Save to same location twice
            var model = TPTestHelpers.CreateSimpleTPMLP(10, 20, 5, context: tpContext);
            await TPCheckpointManager.SaveDistributedAsync(model, checkpointDir);

            var model2 = TPTestHelpers.CreateSimpleTPMLP(10, 20, 5, context: tpContext);
            await TPCheckpointManager.SaveDistributedAsync(model2, checkpointDir);

            // Assert: Should complete without errors
            var checkpoints = TPCheckpointManager.ListCheckpoints(new DirectoryInfo(checkpointDir).Parent.FullName);
            Assert.AreEqual(1, checkpoints.Count);

            // Cleanup
            Directory.Delete(new DirectoryInfo(checkpointDir).Parent.FullName, recursive: true);
        }
    }
}
