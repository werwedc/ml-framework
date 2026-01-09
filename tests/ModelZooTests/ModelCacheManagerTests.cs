using System;
using System.IO;
using System.Linq;
using MLFramework.ModelZoo;
using Xunit;

namespace ModelZooTests
{
    /// <summary>
    /// Unit tests for the ModelCacheManager class.
    /// </summary>
    public class ModelCacheManagerTests : IDisposable
    {
        private readonly string _testCachePath;
        private readonly ModelCacheManager _cacheManager;

        public ModelCacheManagerTests()
        {
            // Create a temporary test cache directory
            _testCachePath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
            var config = new CacheConfiguration
            {
                CacheRootPath = _testCachePath,
                MaxCacheSizeBytes = 100 * 1024 * 1024, // 100MB
                MaxFileAge = TimeSpan.FromDays(30),
                CleanupOnStartup = false
            };
            _cacheManager = new ModelCacheManager(config);
        }

        public void Dispose()
        {
            // Clean up test cache directory
            if (Directory.Exists(_testCachePath))
            {
                try
                {
                    Directory.Delete(_testCachePath, true);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }

            _cacheManager.Dispose();
        }

        [Fact]
        public void GetCachePath_ReturnsConfiguredPath()
        {
            // Act
            string cachePath = _cacheManager.GetCachePath();

            // Assert
            Assert.Equal(_testCachePath, cachePath);
        }

        [Fact]
        public void GetModelPath_ReturnsCorrectPath()
        {
            // Arrange
            string modelName = "resnet50";
            string version = "v1.0.0";
            string expectedPath = Path.Combine(_testCachePath, "models", modelName, version, "model.bin");

            // Act
            string modelPath = _cacheManager.GetModelPath(modelName, version);

            // Assert
            Assert.Equal(expectedPath, modelPath);
        }

        [Fact]
        public void CacheExists_ReturnsFalseForNonExistentModel()
        {
            // Act
            bool exists = _cacheManager.CacheExists("nonexistent", "v1.0.0");

            // Assert
            Assert.False(exists);
        }

        [Fact]
        public void AddToCache_CreatesModelFileAndMetadata()
        {
            // Arrange
            string modelName = "test-model";
            string version = "v1.0.0";
            byte[] modelData = new byte[] { 1, 2, 3, 4, 5 };
            using var stream = new MemoryStream(modelData);

            // Act
            string cachedPath = _cacheManager.AddToCache(modelName, version, stream);

            // Assert
            Assert.True(File.Exists(cachedPath));
            Assert.True(_cacheManager.CacheExists(modelName, version));
            Assert.True(File.Exists(_cacheManager.GetMetadataPath(modelName, version)));
        }

        [Fact]
        public void GetMetadata_ReturnsCorrectMetadata()
        {
            // Arrange
            string modelName = "test-model";
            string version = "v1.0.0";
            byte[] modelData = new byte[] { 1, 2, 3, 4, 5 };
            using var stream = new MemoryStream(modelData);
            _cacheManager.AddToCache(modelName, version, stream);

            // Act
            var metadata = _cacheManager.GetMetadata(modelName, version);

            // Assert
            Assert.NotNull(metadata);
            Assert.Equal(modelName, metadata.ModelName);
            Assert.Equal(version, metadata.Version);
            Assert.Equal(5, metadata.FileSize);
            Assert.Equal(1, metadata.AccessCount);
        }

        [Fact]
        public void GetCacheSize_ReturnsCorrectSize()
        {
            // Arrange
            string modelName = "test-model";
            string version = "v1.0.0";
            byte[] modelData = new byte[1000];
            using var stream = new MemoryStream(modelData);
            _cacheManager.AddToCache(modelName, version, stream);

            // Act
            long cacheSize = _cacheManager.GetCacheSize();

            // Assert
            Assert.True(cacheSize > 0);
        }

        [Fact]
        public void GetCacheSizeFormatted_ReturnsHumanReadableSize()
        {
            // Arrange
            string modelName = "test-model";
            string version = "v1.0.0";
            byte[] modelData = new byte[1000];
            using var stream = new MemoryStream(modelData);
            _cacheManager.AddToCache(modelName, version, stream);

            // Act
            string formattedSize = _cacheManager.GetCacheSizeFormatted();

            // Assert
            Assert.NotNull(formattedSize);
            Assert.Contains("KB", formattedSize);
        }

        [Fact]
        public void ListCachedModels_ReturnsAllCachedModels()
        {
            // Arrange
            _cacheManager.AddToCache("model1", "v1.0.0", new MemoryStream(new byte[100]));
            _cacheManager.AddToCache("model2", "v1.0.0", new MemoryStream(new byte[100]));
            _cacheManager.AddToCache("model1", "v2.0.0", new MemoryStream(new byte[100]));

            // Act
            var models = _cacheManager.ListCachedModels();

            // Assert
            Assert.Equal(3, models.Count);
            Assert.Contains(models, m => m.ModelName == "model1" && m.Version == "v1.0.0");
            Assert.Contains(models, m => m.ModelName == "model2" && m.Version == "v1.0.0");
            Assert.Contains(models, m => m.ModelName == "model1" && m.Version == "v2.0.0");
        }

        [Fact]
        public void RecordAccess_UpdatesLastAccessedAndAccessCount()
        {
            // Arrange
            string modelName = "test-model";
            string version = "v1.0.0";
            _cacheManager.AddToCache(modelName, version, new MemoryStream(new byte[100]));
            var initialMetadata = _cacheManager.GetMetadata(modelName, version);
            DateTime initialAccess = initialMetadata.LastAccessed;
            int initialCount = initialMetadata.AccessCount;

            // Act
            _cacheManager.RecordAccess(modelName, version);

            // Assert
            var updatedMetadata = _cacheManager.GetMetadata(modelName, version);
            Assert.True(updatedMetadata.LastAccessed > initialAccess);
            Assert.Equal(initialCount + 1, updatedMetadata.AccessCount);
        }

        [Fact]
        public void RemoveFromCache_RemovesModelAndMetadata()
        {
            // Arrange
            string modelName = "test-model";
            string version = "v1.0.0";
            _cacheManager.AddToCache(modelName, version, new MemoryStream(new byte[100]));

            // Act
            bool removed = _cacheManager.RemoveFromCache(modelName, version);

            // Assert
            Assert.True(removed);
            Assert.False(_cacheManager.CacheExists(modelName, version));
            Assert.False(File.Exists(_cacheManager.GetModelPath(modelName, version)));
        }

        [Fact]
        public void RemoveFromCache_ReturnsFalseForNonExistentModel()
        {
            // Act
            bool removed = _cacheManager.RemoveFromCache("nonexistent", "v1.0.0");

            // Assert
            Assert.False(removed);
        }

        [Fact]
        public void ClearCache_RemovesAllModels()
        {
            // Arrange
            _cacheManager.AddToCache("model1", "v1.0.0", new MemoryStream(new byte[100]));
            _cacheManager.AddToCache("model2", "v1.0.0", new MemoryStream(new byte[100]));

            // Act
            _cacheManager.ClearCache();

            // Assert
            var models = _cacheManager.ListCachedModels();
            Assert.Empty(models);
            Assert.Equal(0, _cacheManager.Statistics.TotalModels);
        }

        [Fact]
        public void PruneOldModels_RemovesOldModels()
        {
            // Arrange
            // Add a model and manually set its download date to old
            _cacheManager.AddToCache("old-model", "v1.0.0", new MemoryStream(new byte[100]));
            var metadata = _cacheManager.GetMetadata("old-model", "v1.0.0");
            metadata.DownloadDate = DateTime.UtcNow.AddDays(-60);
            _cacheManager.RemoveFromCache("old-model", "v1.0.0");

            // Manually create old model
            string versionDir = Path.Combine(_testCachePath, "models", "old-model", "v1.0.0");
            Directory.CreateDirectory(versionDir);
            File.WriteAllText(Path.Combine(versionDir, "model.bin"), "test data");
            File.WriteAllText(Path.Combine(versionDir, "metadata.json"), metadata.ToJson());

            // Add a recent model
            _cacheManager.AddToCache("new-model", "v1.0.0", new MemoryStream(new byte[100]));

            // Act
            int removed = _cacheManager.PruneOldModels(TimeSpan.FromDays(30));

            // Assert
            Assert.Equal(1, removed);
            Assert.False(_cacheManager.CacheExists("old-model", "v1.0.0"));
            Assert.True(_cacheManager.CacheExists("new-model", "v1.0.0"));
        }

        [Fact]
        public void EnforceMaxCacheSize_RemovesLRUModels()
        {
            // Arrange
            var config = new CacheConfiguration
            {
                CacheRootPath = _testCachePath,
                MaxCacheSizeBytes = 300, // Very small limit
                MaxFileAge = TimeSpan.FromDays(30),
                CleanupOnStartup = false
            };
            var cacheManager = new ModelCacheManager(config);

            // Add three models, each 100 bytes
            cacheManager.AddToCache("model1", "v1.0.0", new MemoryStream(new byte[100]));
            System.Threading.Thread.Sleep(10); // Small delay to ensure different timestamps
            cacheManager.AddToCache("model2", "v1.0.0", new MemoryStream(new byte[100]));
            System.Threading.Thread.Sleep(10);
            cacheManager.AddToCache("model3", "v1.0.0", new MemoryStream(new byte[100]));

            // Access model3 to make it MRU
            cacheManager.RecordAccess("model3", "v1.0.0");

            // Act
            int removed = cacheManager.EnforceMaxCacheSize(300);

            // Assert
            Assert.Equal(1, removed);
            Assert.False(cacheManager.CacheExists("model1", "v1.0.0")); // LRU should be removed
            Assert.False(cacheManager.CacheExists("model2", "v1.0.0")); // Should also be removed to fit limit
            Assert.True(cacheManager.CacheExists("model3", "v1.0.0")); // MRU should remain

            cacheManager.Dispose();
        }

        [Fact]
        public void Statistics_TrackHitsAndMisses()
        {
            // Arrange
            _cacheManager.AddToCache("model1", "v1.0.0", new MemoryStream(new byte[100]));

            // Act
            _cacheManager.RecordAccess("model1", "v1.0.0"); // Hit
            _cacheManager.RecordAccess("model1", "v1.0.0"); // Hit
            _cacheManager.RecordAccess("nonexistent", "v1.0.0"); // Miss

            // Assert
            Assert.Equal(2, _cacheManager.Statistics.CacheHits);
            Assert.Equal(3, _cacheManager.Statistics.TotalLoads);
            Assert.Equal(2.0 / 3.0, _cacheManager.Statistics.CacheHitRate, 2);
        }

        [Fact]
        public void AcquireFileLock_AcquiresAndReleasesLock()
        {
            // Arrange
            string testFile = Path.Combine(_testCachePath, "test.lock");

            // Act
            using (var lock1 = _cacheManager.AcquireFileLock(testFile))
            {
                Assert.True(File.Exists(testFile + ".lock"));

                // Try to acquire another lock (should timeout or throw)
                Assert.Throws<TimeoutException>(() =>
                {
                    using var lock2 = _cacheManager.AcquireFileLock(testFile);
                });
            }

            // Assert - lock file should be cleaned up
            Assert.False(File.Exists(testFile + ".lock"));
        }

        [Fact]
        public void CacheStatistics_FormatBytes_FormatsCorrectly()
        {
            // Act & Assert
            Assert.Equal("0 B", CacheStatistics.FormatBytes(0));
            Assert.Equal("512 B", CacheStatistics.FormatBytes(512));
            Assert.Equal("1 KB", CacheStatistics.FormatBytes(1024));
            Assert.Equal("1.5 KB", CacheStatistics.FormatBytes(1536));
            Assert.Equal("1 MB", CacheStatistics.FormatBytes(1024 * 1024));
            Assert.Equal("1 GB", CacheStatistics.FormatBytes(1024 * 1024 * 1024));
        }

        [Fact]
        public void CacheMetadata_Serialization_RoundTrip()
        {
            // Arrange
            var originalMetadata = new CacheMetadata
            {
                ModelName = "test-model",
                Version = "v1.0.0",
                FileSize = 1024,
                DownloadDate = DateTime.UtcNow,
                LastAccessed = DateTime.UtcNow,
                AccessCount = 5,
                Checksum = "abc123"
            };

            // Act
            string json = originalMetadata.ToJson();
            var deserializedMetadata = CacheMetadata.FromJson(json);

            // Assert
            Assert.Equal(originalMetadata.ModelName, deserializedMetadata.ModelName);
            Assert.Equal(originalMetadata.Version, deserializedMetadata.Version);
            Assert.Equal(originalMetadata.FileSize, deserializedMetadata.FileSize);
            Assert.Equal(originalMetadata.AccessCount, deserializedMetadata.AccessCount);
            Assert.Equal(originalMetadata.Checksum, deserializedMetadata.Checksum);
        }

        [Fact]
        public void CacheConfiguration_DefaultValues_AreReasonable()
        {
            // Act
            var config = new CacheConfiguration();

            // Assert
            Assert.NotNull(config.CacheRootPath);
            Assert.Equal(10L * 1024 * 1024 * 1024, config.MaxCacheSizeBytes); // 10GB
            Assert.Equal(TimeSpan.FromDays(30), config.MaxFileAge);
            Assert.True(config.CleanupOnStartup);
            Assert.Equal(5000, config.LockTimeoutMs);
        }

        [Fact]
        public void MostRecentlyUsedAndLeastRecentlyUsed_AreTracked()
        {
            // Arrange
            _cacheManager.AddToCache("model1", "v1.0.0", new MemoryStream(new byte[100]));
            System.Threading.Thread.Sleep(10);
            _cacheManager.AddToCache("model2", "v1.0.0", new MemoryStream(new byte[100]));
            System.Threading.Thread.Sleep(10);
            _cacheManager.AddToCache("model3", "v1.0.0", new MemoryStream(new byte[100]));

            // Access model3 multiple times
            _cacheManager.RecordAccess("model3", "v1.0.0");
            System.Threading.Thread.Sleep(10);
            _cacheManager.RecordAccess("model3", "v1.0.0");

            // Act
            var mru = _cacheManager.Statistics.MostRecentlyUsed;
            var lru = _cacheManager.Statistics.LeastRecentlyUsed;

            // Assert
            Assert.True(mru.Count > 0);
            Assert.True(lru.Count > 0);
            Assert.Equal("model3", mru[0].ModelName); // Most recently accessed
            Assert.Equal(2, mru[0].AccessCount);
        }

        [Fact]
        public void AddToCache_WithSameVersion_OverwritesOldVersion()
        {
            // Arrange
            string modelName = "test-model";
            string version = "v1.0.0";
            byte[] oldData = new byte[] { 1, 2, 3 };
            byte[] newData = new byte[] { 4, 5, 6, 7 };

            _cacheManager.AddToCache(modelName, version, new MemoryStream(oldData));
            var oldMetadata = _cacheManager.GetMetadata(modelName, version);

            // Act
            _cacheManager.AddToCache(modelName, version, new MemoryStream(newData));

            // Assert
            var newMetadata = _cacheManager.GetMetadata(modelName, version);
            Assert.Equal(4, newMetadata.FileSize);
            Assert.True(newMetadata.DownloadDate > oldMetadata.DownloadDate);
        }
    }
}
