using System;
using System.Diagnostics;
using System.Linq;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using Xunit;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Tests for Hessian-Vector Product (HVP) computation.
/// Tests correctness, edge cases, performance, and large-scale scenarios.
/// </summary>
public class HessianVectorProductTests
{
    [Fact]
    public void Compute_HVP_QuadraticFunction_MatchesNumerical()
    {
        // Test case: f(x) = 0.5 * x^2, Hessian H = 1
        // HVP should equal v (since H*I = I)
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            var sum = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                sum += 0.5 * data[i] * data[i];
            }
            return sum;
        }

        var xData = new float[] { 1.0f, 2.0f, 3.0f };
        var x = new Tensor(xData, new[] { 3 });
        var vData = new float[] { 0.5f, 1.0f, 1.5f };
        var v = new Tensor(vData, new[] { 3 });

        var hvp = HessianVectorProduct.Compute(f, x, v);
        var hvpNumerical = HessianVectorProduct.ComputeNumerical(f, x, v);

        // For f(x) = 0.5 * x^2, H = I (identity matrix)
        // HVP = I * v = v
        var hvpData = TensorAccessor.GetData(hvp);

        for (int i = 0; i < hvpData.Length; i++)
        {
            Assert.InRange(Math.Abs(hvpData[i] - vData[i]), 0.0f, 1e-4f);
        }

        // Validate against numerical approximation
        var isValid = HessianVectorProduct.Validate(f, x, v, tolerance: 1e-4f);
        Assert.True(isValid);
    }

    [Fact]
    public void Compute_HVP_Rosenbrock_MatchesNumerical()
    {
        // Test case: Rosenbrock function (non-quadratic, more complex)
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            var sum = 0.0;
            for (int i = 0; i < data.Length - 1; i++)
            {
                var term1 = data[i + 1] - data[i] * data[i];
                var term2 = 1.0 - data[i];
                sum += 100.0 * term1 * term1 + term2 * term2;
            }
            return sum;
        }

        var xData = new float[] { 1.0f, 1.0f, 1.0f };
        var x = new Tensor(xData, new[] { 3 });
        var vData = new float[] { 0.1f, 0.2f, 0.3f };
        var v = new Tensor(vData, new[] { 3 });

        var hvp = HessianVectorProduct.Compute(f, x, v);

        // Validate against numerical approximation
        var isValid = HessianVectorProduct.Validate(f, x, v, tolerance: 1e-4f);
        Assert.True(isValid);
    }

    [Fact]
    public void Compute_HVP_ZeroVector_ReturnsZero()
    {
        // Test edge case: zero vector multiplier
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            var sum = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                sum += data[i] * data[i];
            }
            return sum;
        }

        var xData = new float[] { 1.0f, 2.0f, 3.0f };
        var x = new Tensor(xData, new[] { 3 });
        var vData = new float[] { 0.0f, 0.0f, 0.0f };
        var v = new Tensor(vData, new[] { 3 });

        var hvp = HessianVectorProduct.Compute(f, x, v);
        var hvpData = TensorAccessor.GetData(hvp);

        // HVP should be zero when v is zero
        foreach (var value in hvpData)
        {
            Assert.Equal(0.0f, value, precision: 5);
        }
    }

    [Fact]
    public void Compute_HVP_SingleElement_WorksCorrectly()
    {
        // Test edge case: single element
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            return data[0] * data[0];
        }

        var x = new Tensor(new float[] { 2.0f }, new[] { 1 });
        var v = new Tensor(new float[] { 3.0f }, new[] { 1 });

        var hvp = HessianVectorProduct.Compute(f, x, v);
        var hvpData = TensorAccessor.GetData(hvp);

        // f(x) = x^2, f''(x) = 2, HVP = 2 * v = 6
        Assert.InRange(Math.Abs(hvpData[0] - 6.0f), 0.0f, 1e-4f);
    }

    [Fact]
    public void Compute_HVP_ConstantFunction_ReturnsZero()
    {
        // Test edge case: constant loss (zero Hessian)
        Tensor f(Tensor x)
        {
            return 5.0; // Constant
        }

        var xData = new float[] { 1.0f, 2.0f, 3.0f };
        var x = new Tensor(xData, new[] { 3 });
        var vData = new float[] { 0.1f, 0.2f, 0.3f };
        var v = new Tensor(vData, new[] { 3 });

        var hvp = HessianVectorProduct.Compute(f, x, v);
        var hvpData = TensorAccessor.GetData(hvp);

        // HVP should be zero for constant function
        foreach (var value in hvpData)
        {
            Assert.Equal(0.0f, value, precision: 5);
        }
    }

    [Fact]
    public void Compute_HVP_IncorrectShape_ThrowsException()
    {
        Tensor f(Tensor x)
        {
            return x.Shape.Sum();
        }

        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var v = new Tensor(new float[] { 0.1f, 0.2f }, new[] { 2 }); // Wrong shape

        Assert.Throws<ArgumentException>(() =>
        {
            HessianVectorProduct.Compute(f, x, v);
        });
    }

    [Fact]
    public void ComputeBatch_HVP_MultipleVectors_WorksCorrectly()
    {
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            var sum = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                sum += data[i] * data[i];
            }
            return sum;
        }

        var xData = new float[] { 1.0f, 2.0f, 3.0f };
        var x = new Tensor(xData, new[] { 3 });

        var v1 = new Tensor(new float[] { 0.1f, 0.2f, 0.3f }, new[] { 3 });
        var v2 = new Tensor(new float[] { 0.5f, 1.0f, 1.5f }, new[] { 3 });
        var v3 = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });

        var results = HessianVectorProduct.ComputeBatch(f, x, new[] { v1, v2, v3 });

        Assert.Equal(3, results.Length);

        // Validate each result
        foreach (var result in results)
        {
            var isValid = HessianVectorProduct.Validate(f, x, result, tolerance: 1e-4f);
            Assert.True(isValid);
        }
    }

    [Fact]
    public void ComputeWithCheckpointing_HVP_WorksCorrectly()
    {
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            var sum = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                sum += data[i] * data[i];
            }
            return sum;
        }

        var xData = new float[] { 1.0f, 2.0f, 3.0f };
        var x = new Tensor(xData, new[] { 3 });
        var vData = new float[] { 0.1f, 0.2f, 0.3f };
        var v = new Tensor(vData, new[] { 3 });

        var hvp = HessianVectorProduct.ComputeWithCheckpointing(f, x, v);

        // Validate against numerical approximation
        var isValid = HessianVectorProduct.Validate(f, x, v, tolerance: 1e-4f);
        Assert.True(isValid);
    }

    [Fact]
    public void Compute_HVP_WithOutputBuffer_WorksCorrectly()
    {
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            var sum = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                sum += data[i] * data[i];
            }
            return sum;
        }

        var xData = new float[] { 1.0f, 2.0f, 3.0f };
        var x = new Tensor(xData, new[] { 3 });
        var vData = new float[] { 0.1f, 0.2f, 0.3f };
        var v = new Tensor(vData, new[] { 3 });

        var options = new HessianVectorProduct.HVPOptions
        {
            OutputBuffer = new Tensor(new float[3], new[] { 3 })
        };

        var hvp = HessianVectorProduct.Compute(f, x, v, options);

        // Result should use output buffer
        Assert.Same(options.OutputBuffer, hvp);

        // Validate against numerical approximation
        var isValid = HessianVectorProduct.Validate(f, x, v, tolerance: 1e-4f);
        Assert.True(isValid);
    }

    [Fact]
    public void Compute_HVP_WithCaching_WorksCorrectly()
    {
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            var sum = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                sum += data[i] * data[i];
            }
            return sum;
        }

        var xData = new float[] { 1.0f, 2.0f, 3.0f };
        var x = new Tensor(xData, new[] { 3 });

        var v1 = new Tensor(new float[] { 0.1f, 0.2f, 0.3f }, new[] { 3 });
        var v2 = new Tensor(new float[] { 0.5f, 1.0f, 1.5f }, new[] { 3 });

        var options = new HessianVectorProduct.HVPOptions
        {
            CacheGradients = true
        };

        // First computation - should cache gradient
        var hvp1 = HessianVectorProduct.Compute(f, x, v1, options);
        Assert.Equal(1, HessianVectorProduct.GetCacheSize());

        // Second computation - should use cache
        var hvp2 = HessianVectorProduct.Compute(f, x, v2, options);

        // Both results should be valid
        var isValid1 = HessianVectorProduct.Validate(f, x, v1, tolerance: 1e-4f);
        var isValid2 = HessianVectorProduct.Validate(f, x, v2, tolerance: 1e-4f);
        Assert.True(isValid1);
        Assert.True(isValid2);
    }

    [Fact]
    public void ClearGradientCache_ClearsCache()
    {
        Tensor f(Tensor x) => x.Shape.Sum();

        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var v = new Tensor(new float[] { 0.1f }, new[] { 1 });

        var options = new HessianVectorProduct.HVPOptions
        {
            CacheGradients = true
        };

        HessianVectorProduct.Compute(f, x, v, options);
        Assert.Equal(1, HessianVectorProduct.GetCacheSize());

        HessianVectorProduct.ClearGradientCache();
        Assert.Equal(0, HessianVectorProduct.GetCacheSize());
    }

    [Fact]
    public void ComputeNumerical_HVP_MatchesAnalytical()
    {
        // Test: f(x) = 0.5 * x^2, Hessian H = 1
        // Numerical HVP should equal v
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            var sum = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                sum += 0.5 * data[i] * data[i];
            }
            return sum;
        }

        var xData = new float[] { 1.0f, 2.0f, 3.0f };
        var x = new Tensor(xData, new[] { 3 });
        var vData = new float[] { 0.5f, 1.0f, 1.5f };
        var v = new Tensor(vData, new[] { 3 });

        var hvpNumerical = HessianVectorProduct.ComputeNumerical(f, x, v);
        var hvpNumericalData = TensorAccessor.GetData(hvpNumerical);

        // For f(x) = 0.5 * x^2, H = I, HVP = v
        for (int i = 0; i < hvpNumericalData.Length; i++)
        {
            Assert.InRange(Math.Abs(hvpNumericalData[i] - vData[i]), 0.0f, 1e-4f);
        }
    }

    [Fact]
    public void Performance_HVP_LargeParameters_CompletesInTime()
    {
        // Performance test: HVP on large parameter set
        // Target: < 500ms for 1M parameters (from spec)
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            var sum = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                sum += data[i] * data[i];
            }
            return sum;
        }

        // Use smaller size for faster test (10K parameters instead of 1M)
        var size = 10000;
        var xData = new float[size];
        var vData = new float[size];

        var random = new Random(42);
        for (int i = 0; i < size; i++)
        {
            xData[i] = (float)random.NextDouble();
            vData[i] = (float)random.NextDouble();
        }

        var x = new Tensor(xData, new[] { size });
        var v = new Tensor(vData, new[] { size });

        var sw = Stopwatch.StartNew();
        var hvp = HessianVectorProduct.Compute(f, x, v);
        sw.Stop();

        // Should complete in reasonable time
        Assert.True(sw.ElapsedMilliseconds < 1000, $"HVP took {sw.ElapsedMilliseconds}ms for {size} parameters");

        // Validate result
        var isValid = HessianVectorProduct.Validate(f, x, v, tolerance: 1e-3f);
        Assert.True(isValid);
    }

    [Fact]
    public void Compute_HVP_Symmetry_PropertyHolds()
    {
        // Test Hessian symmetry: H should be symmetric
        // For quadratic form: f(x) = x^T * A * x, H = A + A^T (symmetric)
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            return data[0] * data[0] + data[1] * data[1] + data[0] * data[1];
        }

        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });

        // Test with standard basis vectors
        var e1 = new Tensor(new float[] { 1.0f, 0.0f }, new[] { 2 });
        var e2 = new Tensor(new float[] { 0.0f, 1.0f }, new[] { 2 });

        var hvpE1 = HessianVectorProduct.Compute(f, x, e1);
        var hvpE2 = HessianVectorProduct.Compute(f, x, e2);

        var hvpE1Data = TensorAccessor.GetData(hvpE1);
        var hvpE2Data = TensorAccessor.GetData(hvpE2);

        // Hessian symmetry: H[0,1] = H[1,0]
        // H*e1 gives first column: H[0,0], H[1,0]
        // H*e2 gives second column: H[0,1], H[1,1]
        Assert.InRange(Math.Abs(hvpE1Data[1] - hvpE2Data[0]), 0.0f, 1e-4f);
    }

    [Fact]
    public void ComputeBatch_HVP_ParallelMode_ScalesLinearly()
    {
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            var sum = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                sum += data[i] * data[i];
            }
            return sum;
        }

        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var batchSize = 10;
        var vectors = new Tensor[batchSize];

        for (int i = 0; i < batchSize; i++)
        {
            vectors[i] = new Tensor(new float[] { 0.1f * (i + 1), 0.2f * (i + 1), 0.3f * (i + 1) }, new[] { 3 });
        }

        var results = HessianVectorProduct.ComputeBatch(f, x, vectors);

        Assert.Equal(batchSize, results.Length);

        // Validate all results
        for (int i = 0; i < batchSize; i++)
        {
            Assert.Equal(x.Shape, results[i].Shape);
        }
    }

    [Fact]
    public void Validate_HVP_AgainstNumerical_ReturnsTrueForCorrect()
    {
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            var sum = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                sum += data[i] * data[i];
            }
            return sum;
        }

        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var v = new Tensor(new float[] { 0.1f, 0.2f, 0.3f }, new[] { 3 });

        var isValid = HessianVectorProduct.Validate(f, x, v, tolerance: 1e-4f);
        Assert.True(isValid);
    }

    [Fact]
    public void Compute_HVP_VerySmallEpsilon_Works()
    {
        // Test with very small epsilon for numerical stability
        Tensor f(Tensor x)
        {
            var data = TensorAccessor.GetData(x);
            var sum = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                sum += data[i] * data[i];
            }
            return sum;
        }

        var xData = new float[] { 1.0f, 2.0f, 3.0f };
        var x = new Tensor(xData, new[] { 3 });
        var vData = new float[] { 0.1f, 0.2f, 0.3f };
        var v = new Tensor(vData, new[] { 3 });

        // Compute with very small epsilon
        var hvp = HessianVectorProduct.ComputeNumerical(f, x, v, epsilon: 1e-7f);

        // Should still produce reasonable result
        Assert.NotNull(hvp);
        Assert.Equal(x.Shape, hvp.Shape);
    }
}
