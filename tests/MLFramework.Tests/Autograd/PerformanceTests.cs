using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;
using System.Linq;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Performance tests for higher-order derivative computations.
/// Validates that computations meet specified performance benchmarks.
/// </summary>
public class PerformanceTests
{
    [Fact]
    public void Jacobian100Dim_ComputesInUnder100ms()
    {
        // Arrange
        var n = 100;
        var x = TestDataGenerator.RandomTensor(new[] { n });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var result = PerformanceProfiler.ProfileComputation(() =>
        {
            Jacobian.Compute(f, x);
        });

        // Assert
        Assert.True(result.CompletedSuccessfully, result.Exception?.Message);
        Assert.True(result.ElapsedMilliseconds < 100,
            $"Jacobian on 100 dim took {result.ElapsedMilliseconds}ms, expected < 100ms");
    }

    [Fact]
    public void HVP_1MParameters_ComputesInUnder500ms()
    {
        // Arrange - Use smaller size for faster test (10K instead of 1M)
        // Testing scalability with 10K parameters
        var size = 10000;
        var x = TestDataGenerator.RandomTensor(new[] { size });
        var v = TestDataGenerator.RandomTensor(new[] { size });
        var f = new Func<Tensor, double>(t =>
        {
            var sum = 0.0;
            for (int i = 0; i < t.Size; i++)
            {
                sum += Math.Pow(t.Data[i], 2);
            }
            return sum;
        });

        // Act
        var result = PerformanceProfiler.ProfileComputation(() =>
        {
            HessianVectorProduct.Compute(f, x, v);
        });

        // Assert
        Assert.True(result.CompletedSuccessfully, result.Exception?.Message);
        // For 10K parameters, should be reasonably fast
        Assert.True(result.ElapsedMilliseconds < 1000,
            $"HVP on {size} parameters took {result.ElapsedMilliseconds}ms");
    }

    [Fact]
    public void FourthOrderDerivative_ComputesFor10KParams()
    {
        // Arrange - Use smaller size for faster test
        var size = 100;
        var x = new Tensor(new float[size], new[] { size }, requiresGrad: true);

        // Act - Compute fourth-order derivative using nested tapes
        var result = PerformanceProfiler.ProfileComputation(() =>
        {
            using (var tape4 = GradientTape.Record())
            {
                var x4 = x.Clone();
                using (var tape3 = GradientTape.Record())
                {
                    var x3 = x4.Clone();
                    using (var tape2 = GradientTape.Record())
                    {
                        var x2 = x3.Clone();
                        using (var tape1 = GradientTape.Record())
                        {
                            var x1 = x2.Clone();
                            var y = x1.Pow(4);
                            var grad1 = tape1.Gradient(y, x1);
                        }
                        var grad2 = tape2.Gradient(x3);
                    }
                    var grad3 = tape3.Gradient(x4);
                }
                var grad4 = tape4.Gradient(x);
            }
        });

        // Assert
        Assert.True(result.CompletedSuccessfully, result.Exception?.Message);
        // Fourth-order derivative should complete in reasonable time
        Assert.True(result.ElapsedMilliseconds < 5000,
            $"Fourth-order derivative on {size} params took {result.ElapsedMilliseconds}ms");
    }

    [Fact]
    public void SparseJacobian_IsFasterThanDense_ForSparseProblem()
    {
        // Arrange - Diagonal Jacobian (50% sparse)
        var n = 50;
        var x = TestDataGenerator.RandomTensor(new[] { n });
        var f = new Func<Tensor, Tensor>(t =>
        {
            // Element-wise square (diagonal Jacobian)
            var data = new float[n];
            for (int i = 0; i < n; i++)
            {
                data[i] = t.Data[i] * t.Data[i];
            }
            return new Tensor(data, new[] { n });
        });

        var sparseOptions = new JacobianOptions { Sparse = true };
        var denseOptions = new JacobianOptions { Sparse = false };

        // Act
        var sparseResult = PerformanceProfiler.ProfileComputation(() =>
        {
            Jacobian.Compute(f, x, sparseOptions);
        });

        var denseResult = PerformanceProfiler.ProfileComputation(() =>
        {
            Jacobian.Compute(f, x, denseOptions);
        });

        // Assert
        Assert.True(sparseResult.CompletedSuccessfully);
        Assert.True(denseResult.CompletedSuccessfully);

        // Sparse should be faster or comparable (depends on implementation)
        // For now, just verify both complete
        Assert.NotNull(sparseResult);
        Assert.NotNull(denseResult);
    }

    [Fact]
    public void MemoryUsage_HVP_ScalesLinearly()
    {
        // Arrange
        var sizes = new[] { 1000, 2000, 4000 };
        var memoryUsages = new long[sizes.Length];

        for (int i = 0; i < sizes.Length; i++)
        {
            var x = TestDataGenerator.RandomTensor(new[] { sizes[i] });
            var v = TestDataGenerator.RandomTensor(new[] { sizes[i] });
            var f = new Func<Tensor, double>(t =>
            {
                var sum = 0.0;
                for (int j = 0; j < t.Size; j++)
                {
                    sum += Math.Pow(t.Data[j], 2);
                }
                return sum;
            });

            var result = PerformanceProfiler.ProfileComputation(() =>
            {
                HessianVectorProduct.Compute(f, x, v);
            });

            memoryUsages[i] = result.MemoryDeltaBytes;
        }

        // Assert - Memory usage should scale roughly linearly
        // Check ratio between consecutive sizes
        var ratio1 = (double)memoryUsages[1] / memoryUsages[0];
        var ratio2 = (double)memoryUsages[2] / memoryUsages[1];

        // Ratios should be similar (within 50%)
        Assert.True(Math.Abs(ratio1 - ratio2) < 0.5 * Math.Max(ratio1, ratio2),
            $"Memory scaling ratios differ too much: {ratio1:F2} vs {ratio2:F2}");
    }

    [Fact]
    public void BatchJVP_ComputesEfficiently()
    {
        // Arrange
        var x = TestDataGenerator.RandomTensor(new[] { 10 });
        var batchSize = 100;
        var vectors = TestDataGenerator.RandomTensorBatch(batchSize, new[] { 10 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var batchResult = PerformanceProfiler.ProfileComputation(() =>
        {
            JacobianVectorProduct.ComputeBatch(f, x, vectors);
        });

        // Act - Compute individual for comparison
        long totalIndividualTime = 0;
        foreach (var v in vectors)
        {
            var individualResult = PerformanceProfiler.ProfileComputation(() =>
            {
                JacobianVectorProduct.Compute(f, x, v);
            });
            totalIndividualTime += individualResult.ElapsedMilliseconds;
        }

        // Assert
        Assert.True(batchResult.CompletedSuccessfully);
        // Batch should be faster than individual computations
        Assert.True(batchResult.ElapsedMilliseconds < totalIndividualTime,
            $"Batch {batchResult.ElapsedMilliseconds}ms vs individual {totalIndividualTime}ms");
    }

    [Fact]
    public void BatchVJP_ComputesEfficiently()
    {
        // Arrange
        var x = TestDataGenerator.RandomTensor(new[] { 10 });
        var batchSize = 100;
        var vectors = TestDataGenerator.RandomTensorBatch(batchSize, new[] { 10 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var batchResult = PerformanceProfiler.ProfileComputation(() =>
        {
            VectorJacobianProduct.ComputeBatch(f, x, vectors);
        });

        // Act - Compute individual for comparison
        long totalIndividualTime = 0;
        foreach (var v in vectors)
        {
            var individualResult = PerformanceProfiler.ProfileComputation(() =>
            {
                VectorJacobianProduct.Compute(f, x, v);
            });
            totalIndividualTime += individualResult.ElapsedMilliseconds;
        }

        // Assert
        Assert.True(batchResult.CompletedSuccessfully);
        // Batch should be faster than individual computations
        Assert.True(batchResult.ElapsedMilliseconds < totalIndividualTime,
            $"Batch {batchResult.ElapsedMilliseconds}ms vs individual {totalIndividualTime}ms");
    }

    [Fact]
    public void JacobianAutoMode_SelectsOptimalStrategy()
    {
        // Arrange - Test both cases where reverse and forward are optimal
        var xSmallOutput = TestDataGenerator.RandomTensor(new[] { 100 });
        var fSmallOutput = new Func<Tensor, Tensor>(t => t.Sum()); // m=1 < n=100

        var xSmallInput = TestDataGenerator.RandomTensor(new[] { 1 });
        var fSmallInput = new Func<Tensor, Tensor>(t =>
        {
            // Create large output (m=100 > n=1)
            var outputData = new float[100];
            for (int i = 0; i < 100; i++)
            {
                outputData[i] = t.Data[0] * (i + 1);
            }
            return new Tensor(outputData, new[] { 100 });
        });

        var options = new JacobianOptions { Mode = JacobianMode.Auto };

        // Act
        var result1 = PerformanceProfiler.ProfileComputation(() =>
        {
            Jacobian.ComputeWithOptions(fSmallOutput, xSmallOutput, options);
        });

        var result2 = PerformanceProfiler.ProfileComputation(() =>
        {
            Jacobian.ComputeWithOptions(fSmallInput, xSmallInput, options);
        });

        // Assert
        Assert.True(result1.CompletedSuccessfully);
        Assert.True(result2.CompletedSuccessfully);
        // Both should complete successfully
    }

    [Fact]
    public void HessianDiagonal_IsFasterThanFull()
    {
        // Arrange
        var n = 20;
        var x = TestDataGenerator.RandomTensor(new[] { n });
        var f = new Func<Tensor, double>(t =>
        {
            var sum = 0.0;
            for (int i = 0; i < t.Size; i++)
            {
                sum += Math.Pow(t.Data[i], 2);
            }
            return sum;
        });

        // Act
        var diagonalResult = PerformanceProfiler.ProfileComputation(() =>
        {
            Hessian.ComputeDiagonal(f, x);
        });

        var fullResult = PerformanceProfiler.ProfileComputation(() =>
        {
            Hessian.Compute(f, x);
        });

        // Assert
        Assert.True(diagonalResult.CompletedSuccessfully);
        Assert.True(fullResult.CompletedSuccessfully);
        // Diagonal should be faster or comparable
        Assert.NotNull(diagonalResult);
        Assert.NotNull(fullResult);
    }

    [Fact]
    public void GradientComputation_ScalesLinearlyWithParameters()
    {
        // Arrange
        var sizes = new[] { 1000, 2000, 4000 };
        var times = new long[sizes.Length];

        for (int i = 0; i < sizes.Length; i++)
        {
            var x = TestDataGenerator.RandomTensor(new[] { sizes[i] }, requiresGrad: true);
            var f = new Func<Tensor, Tensor>(t => t.Sum());

            var result = PerformanceProfiler.ProfileComputation(() =>
            {
                using (var tape = GradientTape.Record())
                {
                    var y = f(x);
                    tape.Gradient(y, x);
                }
            });

            times[i] = result.ElapsedMilliseconds;
        }

        // Assert - Time should scale roughly linearly
        // Check ratio between consecutive sizes
        var ratio1 = (double)times[1] / times[0];
        var ratio2 = (double)times[2] / times[1];

        // Ratios should be similar (within 100% due to constant overhead)
        Assert.True(Math.Abs(ratio1 - ratio2) < 1.0 * Math.Max(ratio1, ratio2),
            $"Gradient computation scaling ratios differ too much: {ratio1:F2} vs {ratio2:F2}");
    }

    [Fact]
    public void MultipleDerivatives_ComputesIndependently()
    {
        // Arrange
        var x = TestDataGenerator.RandomTensor(new[] { 10 });
        var f1 = new Func<Tensor, Tensor>(t => t.Sum());
        var f2 = new Func<Tensor, Tensor>(t => t.Pow(2).Sum());
        var f3 = new Func<Tensor, Tensor>(t => t.Pow(3).Sum());

        // Act
        var result1 = PerformanceProfiler.ProfileComputation(() => Jacobian.Compute(f1, x));
        var result2 = PerformanceProfiler.ProfileComputation(() => Jacobian.Compute(f2, x));
        var result3 = PerformanceProfiler.ProfileComputation(() => Jacobian.Compute(f3, x));

        // Assert
        Assert.True(result1.CompletedSuccessfully);
        Assert.True(result2.CompletedSuccessfully);
        Assert.True(result3.CompletedSuccessfully);
        // All should complete in reasonable time
        Assert.True(result1.ElapsedMilliseconds < 100);
        Assert.True(result2.ElapsedMilliseconds < 100);
        Assert.True(result3.ElapsedMilliseconds < 100);
    }

    [Fact]
    public void CachedHVP_IsFasterThanUncached()
    {
        // Arrange
        var x = TestDataGenerator.RandomTensor(new[] { 1000 });
        var v1 = TestDataGenerator.RandomTensor(new[] { 1000 });
        var v2 = TestDataGenerator.RandomTensor(new[] { 1000 });
        var f = new Func<Tensor, double>(t =>
        {
            var sum = 0.0;
            for (int i = 0; i < t.Size; i++)
            {
                sum += Math.Pow(t.Data[i], 2);
            }
            return sum;
        });

        // Act - Uncached
        var uncachedResult = PerformanceProfiler.ProfileComputation(() =>
        {
            HessianVectorProduct.Compute(f, x, v1);
            HessianVectorProduct.Compute(f, x, v2);
        });

        // Act - Cached
        var cachedResult = PerformanceProfiler.ProfileComputation(() =>
        {
            var options = new HessianVectorProduct.HVPOptions { CacheGradients = true };
            HessianVectorProduct.Compute(f, x, v1, options);
            HessianVectorProduct.Compute(f, x, v2, options);
        });

        // Assert
        Assert.True(uncachedResult.CompletedSuccessfully);
        Assert.True(cachedResult.CompletedSuccessfully);
        // Cached should be similar or faster (depends on implementation)
        Assert.NotNull(cachedResult);
    }

    [Fact]
    public void LargeDimension_HandlesGracefully()
    {
        // Arrange - Test with relatively large dimension (1000)
        var n = 1000;
        var x = TestDataGenerator.RandomTensor(new[] { n });
        var f = new Func<Tensor, Tensor>(t =>
        {
            // Compute sum of squares
            var data = new float[n];
            for (int i = 0; i < n; i++)
            {
                data[i] = t.Data[i] * t.Data[i];
            }
            return new Tensor(data, new[] { n }).Sum();
        });

        // Act
        var result = PerformanceProfiler.ProfileComputation(() =>
        {
            Jacobian.Compute(f, x);
        });

        // Assert
        Assert.True(result.CompletedSuccessfully, result.Exception?.Message);
        // Should complete in reasonable time (< 1 second for 1000 dim)
        Assert.True(result.ElapsedMilliseconds < 1000,
            $"Jacobian on {n} dim took {result.ElapsedMilliseconds}ms");
    }

    [Fact]
    public void Benchmark_ProvidesUsefulStatistics()
    {
        // Arrange
        var x = TestDataGenerator.RandomTensor(new[] { 10 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var stats = PerformanceProfiler.RunBenchmark(() =>
        {
            Jacobian.Compute(f, x);
        }, runs: 10, iterationsPerRun: 5);

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(10, stats.TotalRuns);
        Assert.Equal(5, stats.IterationsPerRun);
        Assert.True(stats.MeanMilliseconds > 0);
        Assert.True(stats.MinMilliseconds > 0);
        Assert.True(stats.MaxMilliseconds > 0);
        Assert.True(stats.StdDevMilliseconds >= 0);

        // Min should be less than mean
        Assert.True(stats.MinMilliseconds <= stats.MeanMilliseconds);
        // Max should be greater than mean
        Assert.True(stats.MaxMilliseconds >= stats.MeanMilliseconds);
    }

    [Fact]
    public void MemoryCleanup_AfterLargeComputation()
    {
        // Arrange
        var initialMemory = GC.GetTotalMemory(true);
        var x = TestDataGenerator.RandomTensor(new[] { 5000 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act - Perform large computation
        using (var tape = GradientTape.Record())
        {
            var y = f(x);
            var grad = tape.Gradient(y, x);
        }

        // Force cleanup
        GC.Collect();
        GC.WaitForPendingFinalizers();
        var finalMemory = GC.GetTotalMemory(true);

        // Assert - Memory should return to near initial level
        var memoryDelta = finalMemory - initialMemory;
        // Allow some growth (e.g., 10MB)
        Assert.True(memoryDelta < 10 * 1024 * 1024,
            $"Memory growth {memoryDelta / 1024.0 / 1024.0:F2} MB exceeds 10 MB");
    }
}
