using System;
using RitterFramework.Core.LoRA;
using RitterFramework.Core.Tensor;
using MLFramework.LoRA;
using Xunit;

namespace MLFramework.Tests.LoRA
{
    /// <summary>
    /// Unit tests for LoraAdapter
    /// </summary>
    public class LoraAdapterTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesAdapter()
        {
            // Arrange
            var name = "test_adapter";
            var config = new LoraConfig { Rank = 8, Alpha = 16, Dropout = 0.0f };

            // Act
            var adapter = new LoraAdapter(name, config);

            // Assert
            Assert.Equal(name, adapter.Name);
            Assert.NotNull(adapter.Config);
            Assert.NotNull(adapter.Weights);
            Assert.NotNull(adapter.Metadata);
            Assert.Empty(adapter.Weights);
        }

        [Fact]
        public void Constructor_WithNullName_ThrowsArgumentNullException()
        {
            // Arrange
            var config = new LoraConfig();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new LoraAdapter(null, config));
        }

        [Fact]
        public void Constructor_WithNullConfig_ThrowsArgumentNullException()
        {
            // Arrange
            var name = "test_adapter";

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new LoraAdapter(name, null));
        }

        [Fact]
        public void AddModuleWeights_WithValidParameters_AddsWeights()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig { Rank = 8, Alpha = 16 });
            var moduleName = "q_proj";
            var loraA = Tensor.Zeros(new[] { 128, 8 });
            var loraB = Tensor.Zeros(new[] { 8, 64 });

            // Act
            adapter.AddModuleWeights(moduleName, loraA, loraB);

            // Assert
            Assert.True(adapter.Weights.ContainsKey(moduleName));
            Assert.NotNull(adapter.Weights[moduleName].LoraA);
            Assert.NotNull(adapter.Weights[moduleName].LoraB);
        }

        [Fact]
        public void AddModuleWeights_WithNullModuleName_ThrowsArgumentException()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig());
            var loraA = Tensor.Zeros(new[] { 128, 8 });
            var loraB = Tensor.Zeros(new[] { 8, 64 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => adapter.AddModuleWeights(null, loraA, loraB));
        }

        [Fact]
        public void AddModuleWeights_WithNullTensors_ThrowsArgumentNullException()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig());

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => adapter.AddModuleWeights("q_proj", null, Tensor.Zeros(new[] { 8, 64 })));
            Assert.Throws<ArgumentNullException>(() => adapter.AddModuleWeights("q_proj", Tensor.Zeros(new[] { 128, 8 }), null));
        }

        [Fact]
        public void AddModuleWeights_CreatesDeepCopyOfTensors()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig { Rank = 8, Alpha = 16 });
            var moduleName = "q_proj";
            var loraA = Tensor.Zeros(new[] { 128, 8 });
            var loraB = Tensor.Zeros(new[] { 8, 64 });

            // Act
            adapter.AddModuleWeights(moduleName, loraA, loraB);

            // Modify original tensors
            loraA.Data[0] = 1.0f;
            loraB.Data[0] = 1.0f;

            // Assert
            Assert.NotEqual(1.0f, adapter.Weights[moduleName].LoraA.Data[0]);
            Assert.NotEqual(1.0f, adapter.Weights[moduleName].LoraB.Data[0]);
        }

        [Fact]
        public void AddModuleWeights_UpdatesUpdatedAtTimestamp()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig());
            var originalUpdatedAt = adapter.Metadata.UpdatedAt;
            var moduleName = "q_proj";

            // Act
            adapter.AddModuleWeights(moduleName, Tensor.Zeros(new[] { 128, 8 }), Tensor.Zeros(new[] { 8, 64 }));

            // Assert
            Assert.True(adapter.Metadata.UpdatedAt > originalUpdatedAt);
        }

        [Fact]
        public void TryGetModuleWeights_WithExistingModule_ReturnsWeights()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig { Rank = 8, Alpha = 16 });
            var moduleName = "q_proj";
            var loraA = Tensor.Zeros(new[] { 128, 8 });
            var loraB = Tensor.Zeros(new[] { 8, 64 });
            adapter.AddModuleWeights(moduleName, loraA, loraB);

            // Act
            var result = adapter.TryGetModuleWeights(moduleName, out var weights);

            // Assert
            Assert.True(result);
            Assert.NotNull(weights);
            Assert.NotNull(weights.LoraA);
            Assert.NotNull(weights.LoraB);
        }

        [Fact]
        public void TryGetModuleWeights_WithNonExistentModule_ReturnsFalse()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig());

            // Act
            var result = adapter.TryGetModuleWeights("non_existent", out var weights);

            // Assert
            Assert.False(result);
            Assert.Null(weights);
        }

        [Fact]
        public void GetParameterCount_WithMultipleModules_CalculatesCorrectly()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig());
            adapter.AddModuleWeights("q_proj", Tensor.Zeros(new[] { 128, 8 }), Tensor.Zeros(new[] { 8, 64 })); // 128*8 + 8*64 = 1024 + 512 = 1536
            adapter.AddModuleWeights("v_proj", Tensor.Zeros(new[] { 128, 4 }), Tensor.Zeros(new[] { 4, 64 }));  // 128*4 + 4*64 = 512 + 256 = 768

            // Act
            var count = adapter.GetParameterCount();

            // Assert
            Assert.Equal(2304, count);
        }

        [Fact]
        public void GetMemorySize_WithParameters_CalculatesCorrectly()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig());
            adapter.AddModuleWeights("q_proj", Tensor.Zeros(new[] { 128, 8 }), Tensor.Zeros(new[] { 8, 64 }));
            var paramCount = adapter.GetParameterCount();

            // Act
            var memorySize = adapter.GetMemorySize();

            // Assert
            Assert.Equal(paramCount * sizeof(float), memorySize);
        }

        [Fact]
        public void Clone_CreatesIndependentCopy()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig { Rank = 8, Alpha = 16 });
            adapter.AddModuleWeights("q_proj", Tensor.Zeros(new[] { 128, 8 }), Tensor.Zeros(new[] { 8, 64 }));
            adapter.Metadata.SetCustomField("test_key", "test_value");

            // Act
            var clonedAdapter = adapter.Clone();

            // Assert
            Assert.Equal(adapter.Name, clonedAdapter.Name);
            Assert.Equal(adapter.Config.Rank, clonedAdapter.Config.Rank);
            Assert.Equal(adapter.Weights.Count, clonedAdapter.Weights.Count);
            Assert.Equal(adapter.Metadata.CreatedAt, clonedAdapter.Metadata.CreatedAt);
            Assert.Equal("test_value", clonedAdapter.Metadata.CustomFields["test_key"]);

            // Verify weights are deep copies
            adapter.Weights["q_proj"].LoraA.Data[0] = 1.0f;
            Assert.NotEqual(1.0f, clonedAdapter.Weights["q_proj"].LoraA.Data[0]);
        }

        [Fact]
        public void Clone_CanBeModifiedIndependently()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig { Rank = 8, Alpha = 16 });
            adapter.AddModuleWeights("q_proj", Tensor.Zeros(new[] { 128, 8 }), Tensor.Zeros(new[] { 8, 64 }));

            // Act
            var clonedAdapter = adapter.Clone();
            clonedAdapter.Name = "cloned_adapter";
            clonedAdapter.Config.Rank = 4;
            clonedAdapter.AddModuleWeights("v_proj", Tensor.Zeros(new[] { 128, 4 }), Tensor.Zeros(new[] { 4, 64 }));

            // Assert
            Assert.Equal("test_adapter", adapter.Name);
            Assert.Equal(8, adapter.Config.Rank);
            Assert.Single(adapter.Weights);

            Assert.Equal("cloned_adapter", clonedAdapter.Name);
            Assert.Equal(4, clonedAdapter.Config.Rank);
            Assert.Equal(2, clonedAdapter.Weights.Count);
        }

        [Fact]
        public void Metadata_CustomFields_AddAndRetrieve()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig());

            // Act
            adapter.Metadata.SetCustomField("training_dataset", "wikipedia");
            adapter.Metadata.SetCustomField("language", "en");

            var result1 = adapter.Metadata.TryGetCustomField("training_dataset", out var value1);
            var result2 = adapter.Metadata.TryGetCustomField("language", out var value2);
            var result3 = adapter.Metadata.TryGetCustomField("non_existent", out var value3);

            // Assert
            Assert.True(result1);
            Assert.Equal("wikipedia", value1);
            Assert.True(result2);
            Assert.Equal("en", value2);
            Assert.False(result3);
            Assert.Null(value3);
        }

        [Fact]
        public void Metadata_TrainingInfo_CanBeSet()
        {
            // Arrange
            var adapter = new LoraAdapter("test_adapter", new LoraConfig());

            // Act
            adapter.Metadata.BaseModel = "llama-7b";
            adapter.Metadata.TrainingEpochs = 10;
            adapter.Metadata.FinalLoss = 0.123f;

            // Assert
            Assert.Equal("llama-7b", adapter.Metadata.BaseModel);
            Assert.Equal(10, adapter.Metadata.TrainingEpochs);
            Assert.Equal(0.123f, adapter.Metadata.FinalLoss);
        }

        [Fact]
        public void Metadata_CreatedAt_IsSetToUtcNow()
        {
            // Arrange & Act
            var before = DateTime.UtcNow.AddSeconds(-1);
            var adapter = new LoraAdapter("test_adapter", new LoraConfig());
            var after = DateTime.UtcNow.AddSeconds(1);

            // Assert
            Assert.InRange(adapter.Metadata.CreatedAt, before, after);
        }
    }
}
