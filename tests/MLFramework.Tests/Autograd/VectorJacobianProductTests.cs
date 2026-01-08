using System;
using System.Linq;
using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Unit tests for Vector-Jacobian Product (VJP) computation.
/// Tests correctness, edge cases, and performance characteristics.
/// </summary>
public class VectorJacobianProductTests : IDisposable
{
    public void Dispose()
    {
        // Clear cache between tests
        VectorJacobianProduct.ClearGradientCache();
    }

    #region Correctness Tests

    [Fact]
    public void Compute_LinearFunction_MatchesNumerical()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1; // f(x) = 2x + 1, f'(x) = 2
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });
        var v = new Tensor(new float[] { 1.0f, 1.0f, 1.0f }, new int[] { 3 });

        // Act
        var vjp = VectorJacobianProduct.Compute(f, x, v);
        var vjpNumerical = VectorJacobianProduct.ComputeNumerical(f, x, v);

        // Assert
        var vjpData = vjp.Data;
        var vjpNumericalData = vjpNumerical.Data;

        for (int i = 0; i < vjpData.Length; i++)
        {
            Assert.Equal(vjpNumericalData[i], vjpData[i], 5); // 5 decimal places precision
        }
    }

    [Fact]
    public void Compute_QuadraticFunction_MatchesNumerical()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * x; // f(x) = x^2, f'(x) = 2x
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });
        var v = new Tensor(new float[] { 1.0f, 1.0f, 1.0f }, new int[] { 3 });

        // Act
        var vjp = VectorJacobianProduct.Compute(f, x, v);
        var vjpNumerical = VectorJacobianProduct.ComputeNumerical(f, x, v);

        // Assert
        var vjpData = vjp.Data;
        var vjpNumericalData = vjpNumerical.Data;

        for (int i = 0; i < vjpData.Length; i++)
        {
            Assert.Equal(vjpNumericalData[i], vjpData[i], 5);
        }
    }

    [Fact]
    public void Compute_MultiOutputFunction_MatchesNumerical()
    {
        // Arrange
        Func<Tensor, Tensor> f = x =>
        {
            var data = x.Data;
            var newData = new float[data.Length * 2];
            for (int i = 0; i < data.Length; i++)
            {
                newData[2 * i] = data[i]; // y1 = x
                newData[2 * i + 1] = data[i] * 2; // y2 = 2x
            }
            return new Tensor(newData, new int[] { data.Length * 2 });
        };

        var x = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
        var v = new Tensor(new float[] { 1.0f, 0.0f, 1.0f, 0.0f }, new int[] { 4 });

        // Act
        var vjp = VectorJacobianProduct.Compute(f, x, v);
        var vjpNumerical = VectorJacobianProduct.ComputeNumerical(f, x, v);

        // Assert
        var vjpData = vjp.Data;
        var vjpNumericalData = vjpNumerical.Data;

        for (int i = 0; i < vjpData.Length; i++)
        {
            Assert.Equal(vjpNumericalData[i], vjpData[i], 5);
        }
    }

    [Fact]
    public void Validate_LinearFunction_ReturnsTrue()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });
        var v = new Tensor(new float[] { 1.0f, 1.0f, 1.0f }, new int[] { 3 });

        // Act & Assert
        Assert.True(VectorJacobianProduct.Validate(f, x, v));
    }

    [Fact]
    public void Validate_QuadraticFunction_ReturnsTrue()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * x;
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });
        var v = new Tensor(new float[] { 1.0f, 1.0f, 1.0f }, new int[] { 3 });

        // Act & Assert
        Assert.True(VectorJacobianProduct.Validate(f, x, v));
    }

    [Fact]
    public void ComputeBatch_MultipleVectors_ConsistentWithIndividual()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });
        var v1 = new Tensor(new float[] { 1.0f, 0.0f, 0.0f }, new int[] { 3 });
        var v2 = new Tensor(new float[] { 0.0f, 1.0f, 0.0f }, new int[] { 3 });
        var v3 = new Tensor(new float[] { 0.0f, 0.0f, 1.0f }, new int[] { 3 });

        // Act
        var batchResult = VectorJacobianProduct.ComputeBatch(f, x, new[] { v1, v2, v3 });
        var individual1 = VectorJacobianProduct.Compute(f, x, v1);
        var individual2 = VectorJacobianProduct.Compute(f, x, v2);
        var individual3 = VectorJacobianProduct.Compute(f, x, v3);

        // Assert
        Assert.Equal(individual1.Data, batchResult[0].Data);
        Assert.Equal(individual2.Data, batchResult[1].Data);
        Assert.Equal(individual3.Data, batchResult[2].Data);
    }

    [Fact]
    public void ComputeMultiple_MultipleInputs_ComputesGradientsForAll()
    {
        // Arrange
        Func<Tensor[], Tensor> f = inputs => inputs[0] + inputs[1]; // f(x, y) = x + y
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
        var y = new Tensor(new float[] { 3.0f, 4.0f }, new int[] { 2 });
        var v = new Tensor(new float[] { 1.0f, 1.0f }, new int[] { 2 });

        // Act
        var results = VectorJacobianProduct.ComputeMultiple(f, new[] { x, y }, v);

        // Assert
        Assert.Equal(2, results.Length);
        Assert.Equal(new float[] { 1.0f, 1.0f }, results[0].Data); // d/dx = 1
        Assert.Equal(new float[] { 1.0f, 1.0f }, results[1].Data); // d/dy = 1
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Compute_ZeroVector_ReturnsZeroGradient()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });
        var v = new Tensor(new float[] { 0.0f, 0.0f, 0.0f }, new int[] { 3 });

        // Act
        var vjp = VectorJacobianProduct.Compute(f, x, v);

        // Assert
        Assert.All(vjp.Data, value => Assert.Equal(0.0f, value));
    }

    [Fact]
    public void Compute_SingleElementVector_WorksCorrectly()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var v = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Act
        var vjp = VectorJacobianProduct.Compute(f, x, v);

        // Assert
        Assert.Single(vjp.Data);
        Assert.Equal(2.0f, vjp.Data[0], 5); // f'(x) = 2, v^T * J = 1 * 2 = 2
    }

    [Fact]
    public void Compute_SparseVector_ExploitsSparsity()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });
        var sparseV = new Tensor(new float[] { 0.0f, 0.0f, 0.0f }, new int[] { 3 });
        var options = new VectorJacobianProduct.VJPOptions { ExploitSparsity = true };

        // Act
        var vjp = VectorJacobianProduct.Compute(f, x, sparseV, options);

        // Assert
        Assert.All(vjp.Data, value => Assert.Equal(0.0f, value));
    }

    [Fact]
    public void Compute_LargeVectorMultiplier_WorksCorrectly()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var largeVData = Enumerable.Range(0, 1000).Select(i => 1.0f / (i + 1)).ToArray();
        var largeV = new Tensor(largeVData, new int[] { 1000 });

        // Note: This test will fail because output size doesn't match
        // We need a function that produces a large output
        Func<Tensor, Tensor> fLarge = input =>
        {
            var outputData = new float[1000];
            for (int i = 0; i < 1000; i++)
            {
                outputData[i] = input.Data[0] * 2 + 1;
            }
            return new Tensor(outputData, new int[] { 1000 });
        };

        // Act
        var vjp = VectorJacobianProduct.Compute(fLarge, x, largeV);

        // Assert - VJP should be sum(v) * f'(x) = sum(1/(i+1)) * 2
        var expected = largeVData.Sum() * 2.0f;
        Assert.Equal(expected, vjp.Data[0], 2); // 2 decimal places due to floating point accumulation
    }

    [Fact]
    public void Compute_WithCache_CachesGradient()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });
        var v = new Tensor(new float[] { 1.0f, 1.0f, 1.0f }, new int[] { 3 });
        var options = new VectorJacobianProduct.VJPOptions { CacheGradients = true };

        // Act
        VectorJacobianProduct.Compute(f, x, v, options);
        var cacheSize = VectorJacobianProduct.GetCacheSize();

        // Assert
        Assert.Equal(1, cacheSize);

        // Clean up
        VectorJacobianProduct.ClearGradientCache();
    }

    [Fact]
    public void ClearGradientCache_ClearsCache()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var v = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var options = new VectorJacobianProduct.VJPOptions { CacheGradients = true };

        // Act
        VectorJacobianProduct.Compute(f, x, v, options);
        VectorJacobianProduct.ClearGradientCache();
        var cacheSize = VectorJacobianProduct.GetCacheSize();

        // Assert
        Assert.Equal(0, cacheSize);
    }

    [Fact]
    public void Compute_WithOutputBuffer_UsesBuffer()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });
        var v = new Tensor(new float[] { 1.0f, 1.0f, 1.0f }, new int[] { 3 });
        var buffer = Tensor.Zeros(new int[] { 3 });
        var options = new VectorJacobianProduct.VJPOptions { OutputBuffer = buffer };

        // Act
        var result = VectorJacobianProduct.Compute(f, x, v, options);

        // Assert
        Assert.Same(buffer, result);
        Assert.Equal(2.0f, result.Data[0], 5);
        Assert.Equal(2.0f, result.Data[1], 5);
        Assert.Equal(2.0f, result.Data[2], 5);
    }

    [Fact]
    public void Compute_WithParallelBatch_ProducesSameResult()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });
        var v1 = new Tensor(new float[] { 1.0f, 0.0f, 0.0f }, new int[] { 3 });
        var v2 = new Tensor(new float[] { 0.0f, 1.0f, 0.0f }, new int[] { 3 });
        var v3 = new Tensor(new float[] { 0.0f, 0.0f, 1.0f }, new int[] { 3 });

        var parallelOptions = new VectorJacobianProduct.VJPOptions { EnableParallel = true };
        var sequentialOptions = new VectorJacobianProduct.VJPOptions { EnableParallel = false };

        // Act
        var parallelResult = VectorJacobianProduct.ComputeBatch(f, x, new[] { v1, v2, v3 }, parallelOptions);
        var sequentialResult = VectorJacobianProduct.ComputeBatch(f, x, new[] { v1, v2, v3 }, sequentialOptions);

        // Assert
        Assert.Equal(parallelResult.Length, sequentialResult.Length);
        for (int i = 0; i < parallelResult.Length; i++)
        {
            Assert.Equal(parallelResult[i].Data, sequentialResult[i].Data);
        }
    }

    #endregion

    #region Exception Tests

    [Fact]
    public void Compute_NullFunction_ThrowsArgumentNullException()
    {
        // Arrange
        Func<Tensor, Tensor> f = null!;
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var v = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => VectorJacobianProduct.Compute(f, x, v));
    }

    [Fact]
    public void Compute_NullInputTensor_ThrowsArgumentNullException()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        Tensor x = null!;
        var v = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => VectorJacobianProduct.Compute(f, x, v));
    }

    [Fact]
    public void Compute_NullVector_ThrowsArgumentNullException()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        Tensor v = null!;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => VectorJacobianProduct.Compute(f, x, v));
    }

    [Fact]
    public void Compute_MismatchedVectorShape_ThrowsArgumentException()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
        var v = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 }); // Wrong shape

        // Act & Assert
        Assert.Throws<ArgumentException>(() => VectorJacobianProduct.Compute(f, x, v));
    }

    [Fact]
    public void ComputeBatch_EmptyVectorBatch_ThrowsArgumentException()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        Tensor[] emptyBatch = Array.Empty<Tensor>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => VectorJacobianProduct.ComputeBatch(f, x, emptyBatch));
    }

    [Fact]
    public void ComputeMultiple_EmptyInputs_ThrowsArgumentException()
    {
        // Arrange
        Func<Tensor[], Tensor> f = inputs => inputs[0];
        Tensor[] emptyInputs = Array.Empty<Tensor>();
        var v = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => VectorJacobianProduct.ComputeMultiple(f, emptyInputs, v));
    }

    [Fact]
    public void Compute_WithInvalidOutputBufferShape_ThrowsArgumentException()
    {
        // Arrange
        Func<Tensor, Tensor> f = x => x * 2 + 1;
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });
        var v = new Tensor(new float[] { 1.0f, 1.0f, 1.0f }, new int[] { 3 });
        var invalidBuffer = Tensor.Zeros(new int[] { 5 }); // Wrong shape
        var options = new VectorJacobianProduct.VJPOptions { OutputBuffer = invalidBuffer };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => VectorJacobianProduct.Compute(f, x, v, options));
    }

    #endregion
}
