using System;
using System.IO;
using RitterFramework.Core.LoRA;
using RitterFramework.Core.Tensor;
using MLFramework.LoRA;
using Xunit;

namespace MLFramework.Tests.LoRA
{
    /// <summary>
    /// Unit tests for AdapterSerializer
    /// </summary>
    public class AdapterSerializerTests
    {
        [Fact]
        public void SaveAndLoad_Binary_PreservesAllData()
        {
            // Arrange
            var adapter = CreateTestAdapter();
            var tempPath = Path.GetTempFileName();

            try
            {
                // Act
                AdapterSerializer.Save(adapter, tempPath);
                var loadedAdapter = AdapterSerializer.Load(tempPath);

                // Assert
                Assert.Equal(adapter.Name, loadedAdapter.Name);
                Assert.Equal(adapter.Config.Rank, loadedAdapter.Config.Rank);
                Assert.Equal(adapter.Config.Alpha, loadedAdapter.Config.Alpha);
                Assert.Equal(adapter.Config.Dropout, loadedAdapter.Config.Dropout);
                Assert.Equal(adapter.Config.Bias, loadedAdapter.Config.Bias);
                Assert.Equal(adapter.Config.LoraType, loadedAdapter.Config.LoraType);
                Assert.Equal(adapter.Weights.Count, loadedAdapter.Weights.Count);
                Assert.Equal(adapter.Metadata.CreatedAt, loadedAdapter.Metadata.CreatedAt);
                Assert.Equal(adapter.Metadata.UpdatedAt, loadedAdapter.Metadata.UpdatedAt);
                Assert.Equal(adapter.Metadata.BaseModel, loadedAdapter.Metadata.BaseModel);
                Assert.Equal(adapter.Metadata.TrainingEpochs, loadedAdapter.Metadata.TrainingEpochs);
                Assert.Equal(adapter.Metadata.FinalLoss, loadedAdapter.Metadata.FinalLoss);

                // Verify weights are preserved
                foreach (var kvp in adapter.Weights)
                {
                    Assert.True(loadedAdapter.Weights.ContainsKey(kvp.Key));
                    var loadedWeights = loadedAdapter.Weights[kvp.Key];

                    Assert.Equal(kvp.Value.LoraA.Shape, loadedWeights.LoraA.Shape);
                    Assert.Equal(kvp.Value.LoraB.Shape, loadedWeights.LoraB.Shape);

                    // Verify tensor data matches
                    for (int i = 0; i < kvp.Value.LoraA.Size; i++)
                    {
                        Assert.Equal(kvp.Value.LoraA.Data[i], loadedWeights.LoraA.Data[i]);
                    }
                    for (int i = 0; i < kvp.Value.LoraB.Size; i++)
                    {
                        Assert.Equal(kvp.Value.LoraB.Data[i], loadedWeights.LoraB.Data[i]);
                    }
                }

                // Verify custom metadata fields
                Assert.Equal(adapter.Metadata.CustomFields.Count, loadedAdapter.Metadata.CustomFields.Count);
                foreach (var kvp in adapter.Metadata.CustomFields)
                {
                    Assert.True(loadedAdapter.Metadata.CustomFields.ContainsKey(kvp.Key));
                    Assert.Equal(kvp.Value, loadedAdapter.Metadata.CustomFields[kvp.Key]);
                }
            }
            finally
            {
                // Cleanup
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        }

        [Fact]
        public void SaveAndLoad_Json_PreservesAllData()
        {
            // Arrange
            var adapter = CreateTestAdapter();
            var tempPath = Path.GetTempFileName();

            try
            {
                // Act
                AdapterSerializer.SaveJson(adapter, tempPath);
                var loadedAdapter = AdapterSerializer.LoadJson(tempPath);

                // Assert
                Assert.Equal(adapter.Name, loadedAdapter.Name);
                Assert.Equal(adapter.Config.Rank, loadedAdapter.Config.Rank);
                Assert.Equal(adapter.Config.Alpha, loadedAdapter.Config.Alpha);
                Assert.Equal(adapter.Config.Dropout, loadedAdapter.Config.Dropout);
                Assert.Equal(adapter.Config.Bias, loadedAdapter.Config.Bias);
                Assert.Equal(adapter.Config.LoraType, loadedAdapter.Config.LoraType);
                Assert.Equal(adapter.Config.TargetModules.Length, loadedAdapter.Config.TargetModules.Length);
            }
            finally
            {
                // Cleanup
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        }

        [Fact]
        public void Load_InvalidFormat_ThrowsException()
        {
            // Arrange
            var tempPath = Path.GetTempFileName();
            File.WriteAllBytes(tempPath, new byte[] { 0x00, 0x01, 0x02, 0x03 });

            try
            {
                // Act & Assert
                Assert.Throws<InvalidDataException>(() => AdapterSerializer.Load(tempPath));
            }
            finally
            {
                // Cleanup
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        }

        [Fact]
        public void Load_FileNotFound_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<FileNotFoundException>(() => AdapterSerializer.Load("nonexistent_file.lora"));
        }

        [Fact]
        public void LoadJson_FileNotFound_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<FileNotFoundException>(() => AdapterSerializer.LoadJson("nonexistent_file.json"));
        }

        [Fact]
        public void Save_NullAdapter_ThrowsException()
        {
            // Arrange
            var tempPath = Path.GetTempFileName();

            try
            {
                // Act & Assert
                Assert.Throws<ArgumentNullException>(() => AdapterSerializer.Save(null, tempPath));
            }
            finally
            {
                // Cleanup
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        }

        [Fact]
        public void Save_EmptyPath_ThrowsException()
        {
            // Arrange
            var adapter = CreateTestAdapter();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => AdapterSerializer.Save(adapter, ""));
        }

        [Fact]
        public void SaveJson_NullAdapter_ThrowsException()
        {
            // Arrange
            var tempPath = Path.GetTempFileName();

            try
            {
                // Act & Assert
                Assert.Throws<ArgumentNullException>(() => AdapterSerializer.SaveJson(null, tempPath));
            }
            finally
            {
                // Cleanup
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        }

        [Fact]
        public void Save_CreatesDirectoryIfNotExists()
        {
            // Arrange
            var adapter = CreateTestAdapter();
            var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString(), "test.lora");

            try
            {
                // Act
                AdapterSerializer.Save(adapter, tempPath);

                // Assert
                Assert.True(File.Exists(tempPath));
            }
            finally
            {
                // Cleanup
                if (File.Exists(tempPath))
                {
                    var directory = Path.GetDirectoryName(tempPath);
                    File.Delete(tempPath);
                    if (Directory.Exists(directory))
                        Directory.Delete(directory);
                }
            }
        }

        [Fact]
        public void Load_InvalidVersion_ThrowsException()
        {
            // Arrange
            var tempPath = Path.GetTempFileName();
            using (var stream = File.Create(tempPath))
            using (var writer = new BinaryWriter(stream))
            {
                // Write valid magic bytes
                writer.Write(new byte[] { 0x4C, 0x6F, 0x52, 0x41 });
                // Write invalid version
                writer.Write((short)99);
            }

            try
            {
                // Act & Assert
                Assert.Throws<InvalidDataException>(() => AdapterSerializer.Load(tempPath));
            }
            finally
            {
                // Cleanup
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        }

        [Fact]
        public void SaveAndLoad_AdapterWithCustomFields_PreservesCustomFields()
        {
            // Arrange
            var adapter = CreateTestAdapter();
            adapter.Metadata.SetCustomField("task_type", "summarization");
            adapter.Metadata.SetCustomField("language", "en");
            adapter.Metadata.SetCustomField("dataset", "cnn_dailymail");
            var tempPath = Path.GetTempFileName();

            try
            {
                // Act
                AdapterSerializer.Save(adapter, tempPath);
                var loadedAdapter = AdapterSerializer.Load(tempPath);

                // Assert
                Assert.True(loadedAdapter.Metadata.TryGetCustomField("task_type", out var taskType));
                Assert.Equal("summarization", taskType);
                Assert.True(loadedAdapter.Metadata.TryGetCustomField("language", out var language));
                Assert.Equal("en", language);
                Assert.True(loadedAdapter.Metadata.TryGetCustomField("dataset", out var dataset));
                Assert.Equal("cnn_dailymail", dataset);
            }
            finally
            {
                // Cleanup
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        }

        [Fact]
        public void SaveAndLoad_AdapterWithMultipleModules_PreservesAllModules()
        {
            // Arrange
            var adapter = CreateTestAdapter();
            // Add more modules
            adapter.AddModuleWeights("k_proj", CreateTensor(new[] { 128, 8 }), CreateTensor(new[] { 8, 64 }));
            adapter.AddModuleWeights("v_proj", CreateTensor(new[] { 128, 8 }), CreateTensor(new[] { 8, 64 }));
            adapter.AddModuleWeights("o_proj", CreateTensor(new[] { 64, 8 }), CreateTensor(new[] { 8, 128 }));
            var tempPath = Path.GetTempFileName();

            try
            {
                // Act
                AdapterSerializer.Save(adapter, tempPath);
                var loadedAdapter = AdapterSerializer.Load(tempPath);

                // Assert
                Assert.Equal(4, loadedAdapter.Weights.Count);
                Assert.True(loadedAdapter.Weights.ContainsKey("q_proj"));
                Assert.True(loadedAdapter.Weights.ContainsKey("k_proj"));
                Assert.True(loadedAdapter.Weights.ContainsKey("v_proj"));
                Assert.True(loadedAdapter.Weights.ContainsKey("o_proj"));
            }
            finally
            {
                // Cleanup
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        }

        /// <summary>
        /// Helper method to create a test adapter
        /// </summary>
        private LoraAdapter CreateTestAdapter()
        {
            var config = new LoraConfig(
                rank: 8,
                alpha: 16,
                dropout: 0.1f,
                targetModules: new[] { "q_proj", "v_proj" },
                bias: "none",
                loraType: "default"
            );

            var adapter = new LoraAdapter("test_adapter", config)
            {
                Metadata =
                {
                    BaseModel = "llama-7b",
                    TrainingEpochs = 10,
                    FinalLoss = 0.523f
                }
            };

            // Add test weights
            adapter.AddModuleWeights("q_proj", CreateTensor(new[] { 128, 8 }), CreateTensor(new[] { 8, 64 }));
            adapter.AddModuleWeights("v_proj", CreateTensor(new[] { 128, 8 }), CreateTensor(new[] { 8, 64 }));

            return adapter;
        }

        /// <summary>
        /// Helper method to create a test tensor with random data
        /// </summary>
        private Tensor CreateTensor(int[] shape)
        {
            var size = 1;
            foreach (var dim in shape)
                size *= dim;

            var data = new float[size];
            var random = new Random(42);
            for (int i = 0; i < size; i++)
            {
                data[i] = (float)random.NextDouble();
            }

            return new Tensor(shape, data);
        }
    }
}
