namespace MLFramework.Checkpointing.Tests;

using System;
using Xunit;

/// <summary>
/// Tests for RecomputationCache
/// </summary>
public class RecomputationCacheTests
{
    [Fact]
    public void Constructor_WithValidMaxSize_CreatesCache()
    {
        // Arrange & Act
        var cache = new RecomputationCache(1024 * 1024);

        // Assert
        var stats = cache.GetStats();
        Assert.Equal(1024 * 1024, stats.MaxSizeBytes);
        Assert.Equal(0, stats.CachedItemsCount);
    }

    [Fact]
    public void Constructor_WithInvalidMaxSize_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new RecomputationCache(0));
        Assert.Throws<ArgumentException>(() => new RecomputationCache(-100));
    }

    [Fact]
    public void Add_WithValidTensor_AddsToCache()
    {
        // Arrange
        var cache = new RecomputationCache(1024);
        var tensor = new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 });

        // Act
        cache.Add("layer1", tensor);

        // Assert
        Assert.True(cache.Contains("layer1"));
        var stats = cache.GetStats();
        Assert.Equal(1, stats.CachedItemsCount);
    }

    [Fact]
    public void Add_WithNullLayerId_ThrowsException()
    {
        // Arrange
        var cache = new RecomputationCache(1024);
        var tensor = new Tensor(new float[] { 1 }, new int[] { 1 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            cache.Add(null!, tensor));
    }

    [Fact]
    public void Add_WithNullTensor_ThrowsException()
    {
        // Arrange
        var cache = new RecomputationCache(1024);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            cache.Add("layer1", null!));
    }

    [Fact]
    public void Get_WithCachedTensor_ReturnsTensor()
    {
        // Arrange
        var cache = new RecomputationCache(1024);
        var tensor = new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 });
        cache.Add("layer1", tensor);

        // Act
        var result = cache.Get("layer1");

        // Assert
        Assert.NotNull(result);
        Assert.Equal(tensor.ElementCount, result!.ElementCount);
    }

    [Fact]
    public void Get_WithUncachedTensor_ReturnsNull()
    {
        // Arrange
        var cache = new RecomputationCache(1024);

        // Act
        var result = cache.Get("nonexistent");

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void Get_WithNullLayerId_ThrowsException()
    {
        // Arrange
        var cache = new RecomputationCache(1024);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            cache.Get(null!));
    }

    [Fact]
    public void Contains_WithCachedTensor_ReturnsTrue()
    {
        // Arrange
        var cache = new RecomputationCache(1024);
        var tensor = new Tensor(new float[] { 1 }, new int[] { 1 });
        cache.Add("layer1", tensor);

        // Act
        var result = cache.Contains("layer1");

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void Contains_WithUncachedTensor_ReturnsFalse()
    {
        // Arrange
        var cache = new RecomputationCache(1024);

        // Act
        var result = cache.Contains("nonexistent");

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void Clear_RemovesAllCachedItems()
    {
        // Arrange
        var cache = new RecomputationCache(1024);
        cache.Add("layer1", new Tensor(new float[] { 1 }, new int[] { 1 }));
        cache.Add("layer2", new Tensor(new float[] { 2 }, new int[] { 1 }));

        // Act
        cache.Clear();

        // Assert
        Assert.False(cache.Contains("layer1"));
        Assert.False(cache.Contains("layer2"));
        var stats = cache.GetStats();
        Assert.Equal(0, stats.CachedItemsCount);
    }

    [Fact]
    public void GetStats_ReturnsCorrectStatistics()
    {
        // Arrange
        var cache = new RecomputationCache(1024);
        var tensor1 = new Tensor(new float[] { 1 }, new int[] { 1 });
        var tensor2 = new Tensor(new float[] { 2 }, new int[] { 1 });
        cache.Add("layer1", tensor1);
        cache.Add("layer2", tensor2);

        // Act
        var stats = cache.GetStats();

        // Assert
        Assert.Equal(2, stats.CachedItemsCount);
        Assert.True(stats.CurrentSizeBytes > 0);
        Assert.Equal(1024, stats.MaxSizeBytes);
        Assert.Equal(0, stats.CacheHits);
        Assert.Equal(0, stats.CacheMisses);
    }

    [Fact]
    public void Get_WithCachedItem_IncrementsHitCount()
    {
        // Arrange
        var cache = new RecomputationCache(1024);
        var tensor = new Tensor(new float[] { 1 }, new int[] { 1 });
        cache.Add("layer1", tensor);

        // Act
        cache.Get("layer1");
        var stats = cache.GetStats();

        // Assert
        Assert.Equal(1, stats.CacheHits);
    }

    [Fact]
    public void Get_WithUncachedItem_IncrementsMissCount()
    {
        // Arrange
        var cache = new RecomputationCache(1024);

        // Act
        cache.Get("nonexistent");
        var stats = cache.GetStats();

        // Assert
        Assert.Equal(1, stats.CacheMisses);
    }

    [Fact]
    public void HitRate_CalculatesCorrectly()
    {
        // Arrange
        var cache = new RecomputationCache(1024);
        var tensor = new Tensor(new float[] { 1 }, new int[] { 1 });
        cache.Add("layer1", tensor);

        // Act
        cache.Get("layer1"); // Hit
        cache.Get("layer1"); // Hit
        cache.Get("nonexistent"); // Miss
        var stats = cache.GetStats();

        // Assert
        Assert.Equal(2.0 / 3.0, stats.HitRate);
    }

    [Fact]
    public void Add_WithExceedingSize_EvictsLRUItem()
    {
        // Arrange
        var cache = new RecomputationCache(100); // Small cache
        var tensor1 = new Tensor(new float[] { 1, 2 }, new int[] { 2 }); // ~8 bytes
        var tensor2 = new Tensor(new float[] { 3, 4 }, new int[] { 2 }); // ~8 bytes
        var tensor3 = new Tensor(new float[] { 5, 6 }, new int[] { 2 }); // ~8 bytes

        // Act
        cache.Add("layer1", tensor1);
        cache.Get("layer1"); // Access layer1
        cache.Add("layer2", tensor2);
        cache.Add("layer3", tensor3); // Should evict layer2 (least recently accessed)

        // Assert
        Assert.True(cache.Contains("layer1"));
        Assert.False(cache.Contains("layer2"));
        Assert.True(cache.Contains("layer3"));
    }

    [Fact]
    public void Add_WithExistingLayerId_ReplacesEntry()
    {
        // Arrange
        var cache = new RecomputationCache(1024);
        var tensor1 = new Tensor(new float[] { 1 }, new int[] { 1 });
        var tensor2 = new Tensor(new float[] { 2, 3 }, new int[] { 2 });
        cache.Add("layer1", tensor1);

        // Act
        cache.Add("layer1", tensor2);

        // Assert
        var result = cache.Get("layer1");
        Assert.Equal(2, result!.ElementCount);
    }

    [Fact]
    public void Add_WithOversizedTensor_DoesNotCache()
    {
        // Arrange
        var cache = new RecomputationCache(10); // Very small cache
        var largeTensor = new Tensor(new float[100], new int[] { 100 }); // ~400 bytes

        // Act
        cache.Add("layer1", largeTensor);

        // Assert
        Assert.False(cache.Contains("layer1"));
    }

    [Fact]
    public void Dispose_AfterDispose_ThrowsOnOperations()
    {
        // Arrange
        var cache = new RecomputationCache(1024);
        cache.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() =>
            cache.Add("layer1", new Tensor(new float[] { 1 }, new int[] { 1 })));

        Assert.Throws<ObjectDisposedException>(() =>
            cache.Get("layer1"));

        Assert.Throws<ObjectDisposedException>(() =>
            cache.Contains("layer1"));
    }

    [Fact]
    public void Clear_ResetsStatistics()
    {
        // Arrange
        var cache = new RecomputationCache(1024);
        var tensor = new Tensor(new float[] { 1 }, new int[] { 1 });
        cache.Add("layer1", tensor);
        cache.Get("layer1");
        cache.Get("nonexistent");

        // Act
        cache.Clear();
        var stats = cache.GetStats();

        // Assert
        Assert.Equal(0, stats.CachedItemsCount);
        Assert.Equal(0, stats.CacheHits);
        Assert.Equal(0, stats.CacheMisses);
    }
}
