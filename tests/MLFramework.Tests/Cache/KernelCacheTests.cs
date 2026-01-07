using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using MLFramework.Cache;

namespace MLFramework.Tests.Cache
{
    /// <summary>
    /// Unit tests for ShapeSignature struct.
    /// </summary>
    public class ShapeSignatureTests
    {
        [Fact]
        public void Create_ValidInput_CreatesSignature()
        {
            // Arrange
            var opName = "MatMul";
            var shapes = new List<int[]>
            {
                new[] { 2, 3 },
                new[] { 3, 4 }
            };

            // Act
            var sig = ShapeSignature.Create(opName, shapes);

            // Assert
            Assert.Equal(opName, sig.OperationName);
            Assert.Equal(2, sig.InputShapes.Length);
            Assert.Equal(new[] { 2, 3 }, sig.InputShapes[0]);
            Assert.Equal(new[] { 3, 4 }, sig.InputShapes[1]);
        }

        [Fact]
        public void Create_NullOperationName_ThrowsArgumentException()
        {
            // Arrange
            var shapes = new List<int[]> { new[] { 2, 3 } };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => ShapeSignature.Create(null!, shapes));
        }

        [Fact]
        public void Create_EmptyOperationName_ThrowsArgumentException()
        {
            // Arrange
            var shapes = new List<int[]> { new[] { 2, 3 } };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => ShapeSignature.Create("", shapes));
        }

        [Fact]
        public void Create_NullShapes_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => ShapeSignature.Create("Op", null!));
        }

        [Fact]
        public void Create_NullShapeInList_ThrowsArgumentException()
        {
            // Arrange
            var shapes = new List<int[]> { new[] { 2, 3 }, null! };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => ShapeSignature.Create("Op", shapes));
        }

        [Fact]
        public void Equals_SameSignatures_ReturnsTrue()
        {
            // Arrange
            var sig1 = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 }, new[] { 3, 4 } });
            var sig2 = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 }, new[] { 3, 4 } });

            // Act & Assert
            Assert.True(sig1.Equals(sig2));
        }

        [Fact]
        public void Equals_DifferentOperationNames_ReturnsFalse()
        {
            // Arrange
            var sig1 = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });
            var sig2 = ShapeSignature.Create("Conv2D", new List<int[]> { new[] { 2, 3 } });

            // Act & Assert
            Assert.False(sig1.Equals(sig2));
        }

        [Fact]
        public void Equals_DifferentShapes_ReturnsFalse()
        {
            // Arrange
            var sig1 = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });
            var sig2 = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 4 } });

            // Act & Assert
            Assert.False(sig1.Equals(sig2));
        }

        [Fact]
        public void Equals_DifferentNumberOfShapes_ReturnsFalse()
        {
            // Arrange
            var sig1 = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 }, new[] { 3, 4 } });
            var sig2 = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });

            // Act & Assert
            Assert.False(sig1.Equals(sig2));
        }

        [Fact]
        public void Equals_Null_ReturnsFalse()
        {
            // Arrange
            var sig = ShapeSignature.Create("Op", new List<int[]> { new[] { 2, 3 } });

            // Act & Assert
            Assert.False(sig.Equals((ShapeSignature?)null));
        }

        [Fact]
        public void GetHashCode_SameSignatures_ReturnsSameHashCode()
        {
            // Arrange
            var sig1 = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 }, new[] { 3, 4 } });
            var sig2 = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 }, new[] { 3, 4 } });

            // Act
            var hash1 = sig1.GetHashCode();
            var hash2 = sig2.GetHashCode();

            // Assert
            Assert.Equal(hash1, hash2);
        }

        [Fact]
        public void GetHashCode_DifferentSignatures_ReturnsDifferentHashCodes()
        {
            // Arrange
            var sig1 = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });
            var sig2 = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 4 } });

            // Act
            var hash1 = sig1.GetHashCode();
            var hash2 = sig2.GetHashCode();

            // Assert
            Assert.NotEqual(hash1, hash2);
        }

        [Fact]
        public void ToString_ReturnsFormattedString()
        {
            // Arrange
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 }, new[] { 3, 4 } });

            // Act
            var result = sig.ToString();

            // Assert
            Assert.Equal("MatMul([2, 3], [3, 4])", result);
        }

        [Fact]
        public void EqualityOperator_SameSignatures_ReturnsTrue()
        {
            // Arrange
            var sig1 = ShapeSignature.Create("Op", new List<int[]> { new[] { 2, 3 } });
            var sig2 = ShapeSignature.Create("Op", new List<int[]> { new[] { 2, 3 } });

            // Act & Assert
            Assert.True(sig1 == sig2);
        }

        [Fact]
        public void InequalityOperator_DifferentSignatures_ReturnsTrue()
        {
            // Arrange
            var sig1 = ShapeSignature.Create("Op", new List<int[]> { new[] { 2, 3 } });
            var sig2 = ShapeSignature.Create("Op", new List<int[]> { new[] { 2, 4 } });

            // Act & Assert
            Assert.True(sig1 != sig2);
        }
    }

    /// <summary>
    /// Unit tests for LRUKernelCache class.
    /// </summary>
    public class LRUKernelCacheTests
    {
        [Fact]
        public void Constructor_ValidMaxSize_CreatesCache()
        {
            // Arrange & Act
            var cache = new LRUKernelCache<int>(10);

            // Assert
            Assert.Equal(10, cache.MaxSize);
            Assert.Equal(0, cache.CurrentSize);
        }

        [Fact]
        public void Constructor_InvalidMaxSize_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => new LRUKernelCache<int>(0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new LRUKernelCache<int>(-1));
        }

        [Fact]
        public void SetAndGet_ValidKernel_CachesAndRetrieves()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 }, new[] { 3, 4 } });

            // Act
            cache.Set(sig, 42, 100);
            var result = cache.Get(sig);

            // Assert
            Assert.Equal(42, result);
        }

        [Fact]
        public void Get_NonExistentSignature_ReturnsDefault()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });

            // Act
            var result = cache.Get(sig);

            // Assert
            Assert.Equal(0, result);
        }

        [Fact]
        public void Set_ExistingSignature_UpdatesKernel()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });

            // Act
            cache.Set(sig, 42, 100);
            cache.Set(sig, 99, 200);
            var result = cache.Get(sig);

            // Assert
            Assert.Equal(99, result);
        }

        [Fact]
        public void Contains_ExistingSignature_ReturnsTrue()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });

            // Act
            cache.Set(sig, 42, 100);

            // Assert
            Assert.True(cache.Contains(sig));
        }

        [Fact]
        public void Contains_NonExistentSignature_ReturnsFalse()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });

            // Act & Assert
            Assert.False(cache.Contains(sig));
        }

        [Fact]
        public void Remove_ExistingSignature_RemovesEntry()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });
            cache.Set(sig, 42, 100);

            // Act
            var removed = cache.Remove(sig);

            // Assert
            Assert.True(removed);
            Assert.False(cache.Contains(sig));
            Assert.Equal(0, cache.CurrentSize);
        }

        [Fact]
        public void Remove_NonExistentSignature_ReturnsFalse()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });

            // Act
            var removed = cache.Remove(sig);

            // Assert
            Assert.False(removed);
        }

        [Fact]
        public void Clear_RemovesAllEntries()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            cache.Set(ShapeSignature.Create("Op1", new List<int[]> { new[] { 2, 3 } }), 1, 100);
            cache.Set(ShapeSignature.Create("Op2", new List<int[]> { new[] { 3, 4 } }), 2, 100);
            cache.Set(ShapeSignature.Create("Op3", new List<int[]> { new[] { 4, 5 } }), 3, 100);

            // Act
            cache.Clear();

            // Assert
            Assert.Equal(0, cache.CurrentSize);
            var stats = cache.GetStats();
            Assert.Equal(0, stats.TotalKernels);
        }

        [Fact]
        public void Get_UpdatesAccessTimeAndUseCount()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });
            cache.Set(sig, 42, 100);
            var firstAccess = DateTime.UtcNow;

            // Act
            System.Threading.Thread.Sleep(10); // Small delay to ensure time difference
            cache.Get(sig);
            cache.Get(sig);

            // Assert
            var stats = cache.GetStats();
            Assert.Equal(2, stats.TotalHits);
        }

        [Fact]
        public void GetStats_ReturnsCorrectStatistics()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig1 = ShapeSignature.Create("Op1", new List<int[]> { new[] { 2, 3 } });
            var sig2 = ShapeSignature.Create("Op2", new List<int[]> { new[] { 3, 4 } });

            // Act
            cache.Set(sig1, 1, 150);
            cache.Set(sig2, 2, 250);
            cache.Get(sig1); // Hit
            cache.Get(sig2); // Hit
            cache.Get(ShapeSignature.Create("NonExistent", new List<int[]> { new[] { 1, 2 } })); // Miss

            // Assert
            var stats = cache.GetStats();
            Assert.Equal(2, stats.TotalKernels);
            Assert.Equal(2, stats.TotalHits);
            Assert.Equal(1, stats.TotalMisses);
            Assert.Equal(0.6666666666666666, stats.HitRate, 8); // 2/3
            Assert.Equal(400, stats.TotalCompilationTimeMs);
            Assert.Equal(200.0, stats.AverageCompilationTimeMs);
        }

        [Fact]
        public void GetStats_EmptyCache_ReturnsZeroStats()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);

            // Act
            var stats = cache.GetStats();

            // Assert
            Assert.Equal(0, stats.TotalKernels);
            Assert.Equal(0, stats.TotalHits);
            Assert.Equal(0, stats.TotalMisses);
            Assert.Equal(0.0, stats.HitRate);
            Assert.Equal(0, stats.TotalCompilationTimeMs);
            Assert.Equal(0.0, stats.AverageCompilationTimeMs);
        }

        [Fact]
        public void CacheFull_EvictsLeastRecentlyUsed()
        {
            // Arrange
            var maxSize = 3;
            var cache = new LRUKernelCache<int>(maxSize);
            var sig1 = ShapeSignature.Create("Op1", new List<int[]> { new[] { 1 } });
            var sig2 = ShapeSignature.Create("Op2", new List<int[]> { new[] { 2 } });
            var sig3 = ShapeSignature.Create("Op3", new List<int[]> { new[] { 3 } });
            var sig4 = ShapeSignature.Create("Op4", new List<int[]> { new[] { 4 } });

            // Act - Fill cache
            cache.Set(sig1, 1, 100);
            System.Threading.Thread.Sleep(10);
            cache.Set(sig2, 2, 100);
            System.Threading.Thread.Sleep(10);
            cache.Set(sig3, 3, 100);

            // Access sig2 and sig3 to make sig1 the LRU
            cache.Get(sig2);
            System.Threading.Thread.Sleep(10);
            cache.Get(sig3);

            // Add new entry - should evict sig1
            cache.Set(sig4, 4, 100);

            // Assert
            Assert.False(cache.Contains(sig1)); // sig1 evicted
            Assert.True(cache.Contains(sig2));
            Assert.True(cache.Contains(sig3));
            Assert.True(cache.Contains(sig4));
            Assert.Equal(maxSize, cache.CurrentSize);
        }

        [Fact]
        public void CacheFullWithSameUseCount_EvictsByAccessTime()
        {
            // Arrange
            var maxSize = 2;
            var cache = new LRUKernelCache<int>(maxSize);
            var sig1 = ShapeSignature.Create("Op1", new List<int[]> { new[] { 1 } });
            var sig2 = ShapeSignature.Create("Op2", new List<int[]> { new[] { 2 } });
            var sig3 = ShapeSignature.Create("Op3", new List<int[]> { new[] { 3 } });

            // Act - Fill cache
            cache.Set(sig1, 1, 100);
            System.Threading.Thread.Sleep(10);
            cache.Set(sig2, 2, 100);

            // Add new entry - should evict sig1 (older)
            cache.Set(sig3, 3, 100);

            // Assert
            Assert.False(cache.Contains(sig1));
            Assert.True(cache.Contains(sig2));
            Assert.True(cache.Contains(sig3));
        }

        [Fact]
        public void GetEvictionCandidates_ReturnsCorrectOrder()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig1 = ShapeSignature.Create("Op1", new List<int[]> { new[] { 1 } });
            var sig2 = ShapeSignature.Create("Op2", new List<int[]> { new[] { 2 } });
            var sig3 = ShapeSignature.Create("Op3", new List<int[]> { new[] { 3 } });

            // Act
            cache.Set(sig1, 1, 100);
            System.Threading.Thread.Sleep(10);
            cache.Set(sig2, 2, 100);
            System.Threading.Thread.Sleep(10);
            cache.Set(sig3, 3, 100);

            var candidates = cache.GetEvictionCandidates(2);

            // Assert
            Assert.Equal(2, candidates.Count);
            Assert.Equal(sig1, candidates[0]); // Least recently used
            Assert.Equal(sig2, candidates[1]);
        }

        [Fact]
        public void StatsReset_ResetsAllStatistics()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig1 = ShapeSignature.Create("Op1", new List<int[]> { new[] { 2, 3 } });
            cache.Set(sig1, 1, 100);
            cache.Get(sig1);
            cache.Get(ShapeSignature.Create("NonExistent", new List<int[]> { new[] { 1, 2 } }));

            // Act
            var stats = cache.GetStats();
            stats.Reset();

            // Assert
            Assert.Equal(0, stats.TotalKernels);
            Assert.Equal(0, stats.TotalHits);
            Assert.Equal(0, stats.TotalMisses);
            Assert.Equal(0.0, stats.HitRate);
            Assert.Equal(0, stats.TotalCompilationTimeMs);
            Assert.Equal(0.0, stats.AverageCompilationTimeMs);
        }

        [Fact]
        public void StatsClone_CreatesIndependentCopy()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig1 = ShapeSignature.Create("Op1", new List<int[]> { new[] { 2, 3 } });
            cache.Set(sig1, 1, 100);
            var originalStats = cache.GetStats();

            // Act
            var clonedStats = originalStats.Clone();

            // Modify cache
            cache.Set(ShapeSignature.Create("Op2", new List<int[]> { new[] { 3, 4 } }), 2, 200);

            // Assert
            Assert.Equal(1, originalStats.TotalKernels); // Original unchanged
            Assert.Equal(1, clonedStats.TotalKernels); // Clone unchanged
            Assert.Equal(2, cache.GetStats().TotalKernels); // Cache updated
        }

        [Fact]
        public void Set_NegativeCompilationTime_ThrowsException()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(10);
            var sig = ShapeSignature.Create("Op", new List<int[]> { new[] { 2, 3 } });

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => cache.Set(sig, 1, -1));
        }

        [Fact]
        public void KernelCacheEntry_Properties_AreCorrect()
        {
            // Arrange
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });

            // Act
            var entry = new KernelCacheEntry<int>(sig, 42, 150);

            // Assert
            Assert.Equal(sig, entry.Signature);
            Assert.Equal(42, entry.CompiledKernel);
            Assert.Equal(0, entry.UseCount);
            Assert.Equal(150, entry.CompilationTimeMs);
            Assert.True((DateTime.UtcNow - entry.LastUsed).TotalSeconds < 1);
        }

        [Fact]
        public void KernelCacheEntry_UpdateAccessTime_UpdatesTimestamp()
        {
            // Arrange
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });
            var entry = new KernelCacheEntry<int>(sig, 42, 150);
            var initialTime = entry.LastUsed;

            // Act
            System.Threading.Thread.Sleep(10);
            entry.UpdateAccessTime();

            // Assert
            Assert.True(entry.LastUsed > initialTime);
        }

        [Fact]
        public void KernelCacheEntry_IncrementUseCount_IncrementsCounter()
        {
            // Arrange
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });
            var entry = new KernelCacheEntry<int>(sig, 42, 150);

            // Act
            entry.IncrementUseCount();
            entry.IncrementUseCount();

            // Assert
            Assert.Equal(2, entry.UseCount);
        }

        [Fact]
        public void ToString_ReturnsFormattedString()
        {
            // Arrange
            var sig = ShapeSignature.Create("MatMul", new List<int[]> { new[] { 2, 3 } });
            var entry = new KernelCacheEntry<int>(sig, 42, 150);

            // Act
            var result = entry.ToString();

            // Assert
            Assert.Contains("KernelCacheEntry", result);
            Assert.Contains("MatMul", result);
            Assert.Contains("Uses=0", result);
            Assert.Contains("CompTime=150ms", result);
        }

        [Fact]
        public void ConcurrentAccess_ThreadSafe()
        {
            // Arrange
            var cache = new LRUKernelCache<int>(100);
            var tasks = new List<Task>();
            var exceptions = new System.Collections.Concurrent.ConcurrentQueue<Exception>();

            // Act - Add and get from multiple threads
            for (int i = 0; i < 10; i++)
            {
                int threadId = i;
                tasks.Add(Task.Run(() =>
                {
                    try
                    {
                        for (int j = 0; j < 10; j++)
                        {
                            var sig = ShapeSignature.Create($"Op{threadId}_{j}", new List<int[]> { new[] { j, j + 1 } });
                            cache.Set(sig, threadId * 10 + j, 50);
                            var result = cache.Get(sig);
                            Assert.Equal(threadId * 10 + j, result);
                        }
                    }
                    catch (Exception ex)
                    {
                        exceptions.Enqueue(ex);
                    }
                }));
            }

            Task.WaitAll(tasks.ToArray());

            // Assert
            Assert.Empty(exceptions);
            Assert.Equal(100, cache.CurrentSize);
        }
    }
}
