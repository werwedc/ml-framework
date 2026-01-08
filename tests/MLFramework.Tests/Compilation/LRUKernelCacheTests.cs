using Xunit;
using MLFramework.Compilation;
using MLFramework.Fusion.Dynamic;

namespace MLFramework.Tests.Compilation;

/// <summary>
/// Unit tests for LRUKernelCache
/// </summary>
public class LRUKernelCacheTests
{
    [Fact]
    public void Constructor_WithDefaultMaxSize_CreatesCache()
    {
        // Arrange & Act
        var cache = new LRUKernelCache<CompiledKernel>(100);

        // Assert
        Assert.Equal(100, cache.MaxSize);
        Assert.Equal(0, cache.CurrentSize);
    }

    [Fact]
    public void Get_EmptyCache_ReturnsNull()
    {
        // Arrange
        var cache = new LRUKernelCache<CompiledKernel>(10);
        var signature = ShapeSignature.Create("Add", new List<int[]> { new[] { 32, 128 } });

        // Act
        var result = cache.Get(signature);

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void SetAndGet_ValidKernel_ReturnsKernel()
    {
        // Arrange
        var cache = new LRUKernelCache<CompiledKernel>(10);
        var signature = ShapeSignature.Create("Mul", new List<int[]> { new[] { 64, 256 } });
        var kernel = CreateTestKernel("test-kernel-1");

        // Act
        cache.Set(signature, kernel);
        var result = cache.Get(signature);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("test-kernel-1", result!.KernelId);
        Assert.Equal(1, cache.CurrentSize);
    }

    [Fact]
    public void Contains_ExistingKernel_ReturnsTrue()
    {
        // Arrange
        var cache = new LRUKernelCache<CompiledKernel>(10);
        var signature = ShapeSignature.Create("Relu", new List<int[]> { new[] { 32, 512 } });
        var kernel = CreateTestKernel("relu-kernel");

        // Act
        cache.Set(signature, kernel);
        var result = cache.Contains(signature);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void Contains_NonExistingKernel_ReturnsFalse()
    {
        // Arrange
        var cache = new LRUKernelCache<CompiledKernel>(10);
        var signature = ShapeSignature.Create("Sigmoid", new List<int[]> { new[] { 32, 512 } });

        // Act
        var result = cache.Contains(signature);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void Remove_ExistingKernel_RemovesFromCache()
    {
        // Arrange
        var cache = new LRUKernelCache<CompiledKernel>(10);
        var signature = ShapeSignature.Create("Tanh", new List<int[]> { new[] { 16, 256 } });
        var kernel = CreateTestKernel("tanh-kernel");
        cache.Set(signature, kernel);

        // Act
        cache.Remove(signature);
        var result = cache.Get(signature);

        // Assert
        Assert.Null(result);
        Assert.Equal(0, cache.CurrentSize);
    }

    [Fact]
    public void Clear_Cache_emptiesCache()
    {
        // Arrange
        var cache = new LRUKernelCache<CompiledKernel>(10);
        cache.Set(ShapeSignature.Create("Op1", new List<int[]> { new[] { 32, 128 } }), CreateTestKernel("kernel1"));
        cache.Set(ShapeSignature.Create("Op2", new List<int[]> { new[] { 64, 256 } }), CreateTestKernel("kernel2"));

        // Act
        cache.Clear();

        // Assert
        Assert.Equal(0, cache.CurrentSize);
        Assert.Equal(0, cache.GetStats().TotalKernels);
    }

    [Fact]
    public void GetStats_AfterOperations_ReturnsCorrectStats()
    {
        // Arrange
        var cache = new LRUKernelCache<CompiledKernel>(10);
        var signature1 = ShapeSignature.Create("Op1", new List<int[]> { new[] { 32, 128 } });
        var signature2 = ShapeSignature.Create("Op2", new List<int[]> { new[] { 64, 256 } });

        // Act
        cache.Set(signature1, CreateTestKernel("kernel1"));
        cache.Get(signature1); // Hit
        cache.Get(signature1); // Hit
        cache.Get(signature2); // Miss

        // Assert
        var stats = cache.GetStats();
        Assert.Equal(1, stats.TotalKernels);
        Assert.Equal(2, stats.TotalHits);
        Assert.Equal(1, stats.TotalMisses);
    }

    [Fact]
    public void CacheExceedingMaxSize_EvictsLeastRecentlyUsed()
    {
        // Arrange
        var cache = new LRUKernelCache<CompiledKernel>(3);
        var sig1 = ShapeSignature.Create("Op1", new List<int[]> { new[] { 32 } });
        var sig2 = ShapeSignature.Create("Op2", new List<int[]> { new[] { 64 } });
        var sig3 = ShapeSignature.Create("Op3", new List<int[]> { new[] { 128 } });
        var sig4 = ShapeSignature.Create("Op4", new List<int[]> { new[] { 256 } });

        // Act
        cache.Set(sig1, CreateTestKernel("kernel1"));
        cache.Set(sig2, CreateTestKernel("kernel2"));
        cache.Set(sig3, CreateTestKernel("kernel3"));
        cache.Get(sig1); // Access sig1 to make it recently used
        cache.Set(sig4, CreateTestKernel("kernel4")); // Should evict sig2 (LRU)

        // Assert
        Assert.Equal(3, cache.CurrentSize);
        Assert.NotNull(cache.Get(sig1));
        Assert.Null(cache.Get(sig2)); // sig2 should be evicted
        Assert.NotNull(cache.Get(sig3));
        Assert.NotNull(cache.Get(sig4));
    }

    [Fact]
    public void HitRate_CalculatedCorrectly()
    {
        // Arrange
        var cache = new LRUKernelCache<CompiledKernel>(10);
        var signature = ShapeSignature.Create("Op", new List<int[]> { new[] { 32, 128 } });

        // Act
        cache.Set(signature, CreateTestKernel("kernel"));
        cache.Get(signature); // Hit
        cache.Get(signature); // Hit
        cache.Get(ShapeSignature.Create("OtherOp", new List<int[]> { new[] { 64, 256 } })); // Miss

        // Assert
        var stats = cache.GetStats();
        Assert.Equal(0.6667, stats.HitRate, precision: 4);
    }

    private CompiledKernel CreateTestKernel(string id)
    {
        return new CompiledKernel
        {
            KernelId = id,
            SourceCode = $"kernel {id}",
            Binary = Array.Empty<byte>(),
            SpecializedShapes = new List<int[]>(),
            IsGeneric = false,
            Signature = $"sig-{id}",
            EstimatedExecutionTimeNs = 1000
        };
    }
}
