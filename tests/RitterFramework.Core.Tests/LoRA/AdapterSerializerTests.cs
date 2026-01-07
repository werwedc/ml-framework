using RitterFramework.Core.LoRA;
using RitterFramework.Core.Tensor;
using Xunit;

namespace RitterFramework.Core.Tests.LoRA
{
    public class AdapterSerializerTests
    {
        private LoraAdapter _testAdapter;

        public AdapterSerializerTests()
        {
            _testAdapter = CreateTestAdapter();
        }

        [Fact]
        public void SaveAndLoad_Binary_PreservesAllData()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), "test_adapter_" + Guid.NewGuid() + ".lora");

            AdapterSerializer.Save(_testAdapter, tempPath);
            var loadedAdapter = AdapterSerializer.Load(tempPath);

            Assert.Equal(_testAdapter.Name, loadedAdapter.Name);
            Assert.Equal(_testAdapter.Config.Rank, loadedAdapter.Config.Rank);
            Assert.Equal(_testAdapter.Config.Alpha, loadedAdapter.Config.Alpha);
            Assert.Equal(_testAdapter.Config.Dropout, loadedAdapter.Config.Dropout);
            Assert.Equal(_testAdapter.Weights.Count, loadedAdapter.Weights.Count);
            Assert.Equal(_testAdapter.Metadata.CreatedAt, loadedAdapter.Metadata.CreatedAt);
            Assert.Equal(_testAdapter.Metadata.Creator, loadedAdapter.Metadata.Creator);
            Assert.Equal(_testAdapter.Metadata.Description, loadedAdapter.Metadata.Description);

            File.Delete(tempPath);
        }

        [Fact]
        public void SaveAndLoad_Json_PreservesAllData()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), "test_adapter_" + Guid.NewGuid() + ".json");

            AdapterSerializer.SaveJson(_testAdapter, tempPath);
            var loadedAdapter = AdapterSerializer.LoadJson(tempPath);

            Assert.Equal(_testAdapter.Name, loadedAdapter.Name);
            Assert.Equal(_testAdapter.Config.Rank, loadedAdapter.Config.Rank);
            Assert.Equal(_testAdapter.Config.Alpha, loadedAdapter.Config.Alpha);
            Assert.Equal(_testAdapter.Config.Dropout, loadedAdapter.Config.Dropout);
            Assert.Equal(_testAdapter.Weights.Count, loadedAdapter.Weights.Count);

            File.Delete(tempPath);
        }

        [Fact]
        public void Load_BinaryWithInvalidMagicNumber_ThrowsInvalidDataException()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), "test_adapter_" + Guid.NewGuid() + ".lora");

            using (var writer = new BinaryWriter(File.Open(tempPath, FileMode.Create)))
            {
                writer.Write(0x12345678); // Invalid magic number
                writer.Write(1); // version
            }

            Assert.Throws<InvalidDataException>(() => AdapterSerializer.Load(tempPath));

            File.Delete(tempPath);
        }

        [Fact]
        public void Load_BinaryWithUnsupportedVersion_ThrowsInvalidDataException()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), "test_adapter_" + Guid.NewGuid() + ".lora");

            using (var writer = new BinaryWriter(File.Open(tempPath, FileMode.Create)))
            {
                writer.Write(0x4C4F5241); // Correct magic number
                writer.Write(99); // Unsupported version
            }

            Assert.Throws<InvalidDataException>(() => AdapterSerializer.Load(tempPath));

            File.Delete(tempPath);
        }

        [Fact]
        public void Load_NonExistentFile_ThrowsFileNotFoundException()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), "nonexistent_" + Guid.NewGuid() + ".lora");

            Assert.Throws<FileNotFoundException>(() => AdapterSerializer.Load(tempPath));
        }

        [Fact]
        public void Load_InvalidJson_ThrowsJsonException()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), "test_adapter_" + Guid.NewGuid() + ".json");
            File.WriteAllText(tempPath, "not a valid json");

            Assert.Throws<System.Text.Json.JsonException>(() => AdapterSerializer.LoadJson(tempPath));

            File.Delete(tempPath);
        }

        [Fact]
        public void Save_Binary_CreatesDirectoryIfNotExists()
        {
            var tempDir = Path.Combine(Path.GetTempPath(), "test_" + Guid.NewGuid());
            var tempPath = Path.Combine(tempDir, "test_adapter.lora");

            var exception = Record.Exception(() => AdapterSerializer.Save(_testAdapter, tempPath));
            Assert.Null(exception);
            Assert.True(File.Exists(tempPath));

            Directory.Delete(tempDir, true);
        }

        [Fact]
        public void Save_Json_CreatesDirectoryIfNotExists()
        {
            var tempDir = Path.Combine(Path.GetTempPath(), "test_" + Guid.NewGuid());
            var tempPath = Path.Combine(tempDir, "test_adapter.json");

            var exception = Record.Exception(() => AdapterSerializer.SaveJson(_testAdapter, tempPath));
            Assert.Null(exception);
            Assert.True(File.Exists(tempPath));

            Directory.Delete(tempDir, true);
        }

        [Fact]
        public void Save_NullAdapter_ThrowsArgumentNullException()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), "test_adapter.lora");

            Assert.Throws<ArgumentNullException>(() => AdapterSerializer.Save(null, tempPath));
            Assert.Throws<ArgumentNullException>(() => AdapterSerializer.SaveJson(null, tempPath));
        }

        [Fact]
        public void Save_EmptyPath_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => AdapterSerializer.Save(_testAdapter, ""));
            Assert.Throws<ArgumentException>(() => AdapterSerializer.SaveJson(_testAdapter, ""));
        }

        private LoraAdapter CreateTestAdapter()
        {
            var config = new LoraConfig(rank: 8, alpha: 16, dropout: 0.05f, targetModules: new[] { "layer1", "layer2" });
            var adapter = new LoraAdapter("test_adapter", config);

            adapter.Metadata = new LoraAdapterMetadata
            {
                Creator = "test_user",
                Description = "Test adapter for unit tests",
                Version = "1.0"
            };

            // Add test weights for two modules
            var loraA1 = Core.Tensor.Tensor.Zeros(new[] { 8, 64 });
            var loraB1 = Core.Tensor.Tensor.Zeros(new[] { 128, 8 });
            adapter.AddModuleWeights("layer1", new LoraModuleWeights(loraA1, loraB1));

            var loraA2 = Core.Tensor.Tensor.Zeros(new[] { 8, 32 });
            var loraB2 = Core.Tensor.Tensor.Zeros(new[] { 64, 8 });
            adapter.AddModuleWeights("layer2", new LoraModuleWeights(loraA2, loraB2));

            return adapter;
        }
    }
}
