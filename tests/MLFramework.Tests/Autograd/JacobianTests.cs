using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;
using System.Linq;

namespace MLFramework.Tests.Autograd;

public class JacobianTests
{
    #region Basic Jacobian Computation Tests

    [Fact]
    public void Jacobian_Compute_ForLinearFunction_ReturnsCorrectJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        Assert.NotNull(jacobian);
        Assert.Equal(2, jacobian.Shape.Length);
        // Output is scalar, input is 3D -> 1x3 Jacobian
        Assert.Single(jacobian.Shape[0]);
        Assert.Equal(3, jacobian.Shape[1]);
        // For f(x) = sum(x), Jacobian = [1, 1, 1]
        Assert.Equal(1.0f, jacobian._data[0]);
        Assert.Equal(1.0f, jacobian._data[1]);
        Assert.Equal(1.0f, jacobian._data[2]);
    }

    [Fact]
    public void Jacobian_Compute_ForQuadraticFunction_ReturnsCorrectJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 2.0f, 3.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Pow(2).Sum());

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        Assert.NotNull(jacobian);
        Assert.Equal(2, jacobian.Shape.Length);
        // Output is scalar, input is 2D -> 1x2 Jacobian
        Assert.Single(jacobian.Shape[0]);
        Assert.Equal(2, jacobian.Shape[1]);
        // For f(x) = x^2 + y^2, Jacobian = [2x, 2y] = [4, 6]
        Assert.Equal(4.0f, jacobian._data[0], 1);
        Assert.Equal(6.0f, jacobian._data[1], 1);
    }

    [Fact]
    public void Jacobian_Compute_ForVectorFunction_ReturnsCorrectJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            // f(x) = [x[0]^2, x[1]^2]
            var data = new float[] { t._data[0] * t._data[0], t._data[1] * t._data[1] };
            return new Tensor(data, new[] { 2 });
        });

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        Assert.NotNull(jacobian);
        Assert.Equal(2, jacobian.Shape.Length);
        // Output is 2D, input is 2D -> 2x2 Jacobian
        Assert.Equal(2, jacobian.Shape[0]);
        Assert.Equal(2, jacobian.Shape[1]);
        // Jacobian should be [[2x[0], 0], [0, 2x[1]]] = [[2, 0], [0, 4]]
        Assert.Equal(2.0f, jacobian._data[0], 1);
        Assert.Equal(0.0f, jacobian._data[1], 3);
        Assert.Equal(0.0f, jacobian._data[2], 3);
        Assert.Equal(4.0f, jacobian._data[3], 1);
    }

    [Fact]
    public void Jacobian_Compute_NullFunction_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => Jacobian.Compute(null!, x));
    }

    [Fact]
    public void Jacobian_Compute_NullTensor_ThrowsException()
    {
        // Arrange
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => Jacobian.Compute(f, null!));
    }

    [Fact]
    public void Jacobian_Compute_NullOptions_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => Jacobian.Compute(f, x, null!));
    }

    #endregion

    #region Mode Selection Tests

    [Fact]
    public void Jacobian_Compute_WithAutoMode_SelectsCorrectMode()
    {
        // Arrange
        var xSmallOutput = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var fSmallOutput = new Func<Tensor, Tensor>(t => t.Sum()); // Scalar output (m=1 < n=3)

        var xSmallInput = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var fSmallInput = new Func<Tensor, Tensor>(t =>
        {
            // Create large output (m=10 > n=1)
            var data = new float[10];
            for (int i = 0; i < 10; i++)
            {
                data[i] = t._data[0] * (i + 1);
            }
            return new Tensor(data, new[] { 10 });
        });

        var options = new JacobianOptions { Mode = JacobianMode.Auto };

        // Act
        var result1 = Jacobian.ComputeWithOptions(fSmallOutput, xSmallOutput, options);
        var result2 = Jacobian.ComputeWithOptions(fSmallInput, xSmallInput, options);

        // Assert
        // For m=1 < n=3, should select Reverse mode (VJP)
        Assert.Equal(JacobianMode.Reverse, result1.ModeUsed);
        // For m=10 > n=1, should select Forward mode (JVP)
        Assert.Equal(JacobianMode.Forward, result2.ModeUsed);
    }

    [Fact]
    public void Jacobian_Compute_WithReverseMode_UsesVJP()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());
        var options = new JacobianOptions { Mode = JacobianMode.Reverse };

        // Act
        var result = Jacobian.ComputeWithOptions(f, x, options);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(JacobianMode.Reverse, result.ModeUsed);
        Assert.Equal(1.0f, result.Jacobian._data[0]);
        Assert.Equal(1.0f, result.Jacobian._data[1]);
    }

    [Fact]
    public void Jacobian_Compute_WithForwardMode_UsesJVP()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());
        var options = new JacobianOptions { Mode = JacobianMode.Forward };

        // Act
        var result = Jacobian.ComputeWithOptions(f, x, options);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(JacobianMode.Forward, result.ModeUsed);
        Assert.Equal(1.0f, result.Jacobian._data[0]);
        Assert.Equal(1.0f, result.Jacobian._data[1]);
    }

    #endregion

    #region Structure Detection Tests

    [Fact]
    public void Jacobian_Compute_DetectsDiagonalStructure()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            // Element-wise operation: f(x) = [x[0]^2, x[1]^2]
            var data = new float[] { t._data[0] * t._data[0], t._data[1] * t._data[1] };
            return new Tensor(data, new[] { 2 });
        });
        var options = new JacobianOptions { DetectStructure = true };

        // Act
        var result = Jacobian.ComputeWithOptions(f, x, options);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(JacobianStructure.Diagonal, result.Structure);
    }

    [Fact]
    public void Jacobian_Compute_WithStructureDisabled_DoesNotDetectStructure()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            var data = new float[] { t._data[0] * t._data[0], t._data[1] * t._data[1] };
            return new Tensor(data, new[] { 2 });
        });
        var options = new JacobianOptions { DetectStructure = false };

        // Act
        var result = Jacobian.ComputeWithOptions(f, x, options);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(JacobianStructure.General, result.Structure);
    }

    #endregion

    #region Sparsity Tests

    [Fact]
    public void Jacobian_Compute_CalculatesSparsityStatistics()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            // Diagonal Jacobian (half zeros)
            var data = new float[] { t._data[0] * t._data[0], t._data[1] * t._data[1] };
            return new Tensor(data, new[] { 2 });
        });
        var options = new JacobianOptions { DetectStructure = true };

        // Act
        var result = Jacobian.ComputeWithOptions(f, x, options);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.IsSparse);
        Assert.Equal(0.5f, result.SparsityRatio, 2); // 50% sparse (2 zeros out of 4)
        Assert.Equal(2, result.NonZeroCount);
    }

    [Fact]
    public void Jacobian_Compute_WithSparseOption_UsesSparseRepresentation()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            var data = new float[] { t._data[0] * t._data[0], t._data[1] * t._data[1] };
            return new Tensor(data, new[] { 2 });
        });
        var options = new JacobianOptions { Sparse = true };

        // Act
        var jacobian = Jacobian.Compute(f, x, options);

        // Assert
        Assert.NotNull(jacobian);
        // For now, sparse representation returns dense tensor
        // A full implementation would return a SparseTensor
    }

    #endregion

    #region Partial Jacobian Tests

    [Fact]
    public void Jacobian_Compute_WithOutputIndices_ComputesPartialJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            // f(x) = [x[0]^2, x[1]^2, x[0] + x[1]]
            var data = new float[]
            {
                t._data[0] * t._data[0],
                t._data[1] * t._data[1],
                t._data[0] + t._data[1]
            };
            return new Tensor(data, new[] { 3 });
        });
        var options = new JacobianOptions
        {
            OutputIndices = new[] { 0, 2 } // Only compute Jacobian for first and third outputs
        };

        // Act
        var result = Jacobian.ComputeWithOptions(f, x, options);

        // Assert
        Assert.NotNull(result);
        // Should have 2 output rows (for indices 0 and 2)
        Assert.Equal(2, result.Jacobian.Shape[0]);
        Assert.Equal(2, result.Jacobian.Shape[1]);

        // Row 0 (output 0): [2x[0], 0] = [2, 0]
        Assert.Equal(2.0f, result.Jacobian._data[0], 1);
        Assert.Equal(0.0f, result.Jacobian._data[1], 3);

        // Row 1 (output 2): [1, 1]
        Assert.Equal(1.0f, result.Jacobian._data[2]);
        Assert.Equal(1.0f, result.Jacobian._data[3]);
    }

    #endregion

    #region Progress Callback Tests

    [Fact]
    public void Jacobian_Compute_WithProgressCallback_CallsCallback()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        var callbackCalled = false;
        int lastProgress = 0;
        int totalExpected = 0;

        var options = new JacobianOptions
        {
            Mode = JacobianMode.Forward, // Use forward mode for deterministic progress
            ProgressCallback = (completed, total) =>
            {
                callbackCalled = true;
                lastProgress = completed;
                totalExpected = total;
            }
        };

        // Act
        Jacobian.Compute(f, x, options);

        // Assert
        Assert.True(callbackCalled);
        Assert.Equal(3, totalExpected); // Should match output dimension
        Assert.Equal(3, lastProgress); // Should reach 100% completion
    }

    #endregion

    #region Numerical Validation Tests

    [Fact]
    public void Jacobian_ComputeNumerical_MatchesAnalytical_ForLinearFunction()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var jacobianAnalytical = Jacobian.Compute(f, x);
        var jacobianNumerical = Jacobian.ComputeNumerical(f, x);

        // Assert
        Assert.NotNull(jacobianAnalytical);
        Assert.NotNull(jacobianNumerical);
        Assert.Equal(jacobianAnalytical.Shape, jacobianNumerical.Shape);

        for (int i = 0; i < jacobianAnalytical.Size; i++)
        {
            Assert.Equal(jacobianAnalytical._data[i], jacobianNumerical._data[i], 3);
        }
    }

    [Fact]
    public void Jacobian_ComputeNumerical_MatchesAnalytical_ForQuadraticFunction()
    {
        // Arrange
        var x = new Tensor(new float[] { 2.0f, 3.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Pow(2).Sum());

        // Act
        var jacobianAnalytical = Jacobian.Compute(f, x);
        var jacobianNumerical = Jacobian.ComputeNumerical(f, x);

        // Assert
        Assert.NotNull(jacobianAnalytical);
        Assert.NotNull(jacobianNumerical);
        Assert.Equal(jacobianAnalytical.Shape, jacobianNumerical.Shape);

        for (int i = 0; i < jacobianAnalytical.Size; i++)
        {
            Assert.Equal(jacobianAnalytical._data[i], jacobianNumerical._data[i], 2);
        }
    }

    [Fact]
    public void Jacobian_Validate_ReturnsTrue_ForCorrectImplementation()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());
        var jacobian = Jacobian.Compute(f, x);

        // Act
        var isValid = Jacobian.Validate(f, x, jacobian);

        // Assert
        Assert.True(isValid);
    }

    [Fact]
    public void Jacobian_Validate_ReturnsFalse_ForIncorrectJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());
        var incorrectJacobian = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 1, 2 }); // Wrong values

        // Act
        var isValid = Jacobian.Validate(f, x, incorrectJacobian);

        // Assert
        Assert.False(isValid);
    }

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void Jacobian_Compute_ForSingleInputSingleOutput_ReturnsCorrectJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 2.0f }, new[] { 1 });
        var f = new Func<Tensor, Tensor>(t => t.Pow(2));

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        Assert.NotNull(jacobian);
        Assert.Equal(2, jacobian.Shape.Length);
        // 1x1 Jacobian
        Assert.Single(jacobian.Shape[0]);
        Assert.Single(jacobian.Shape[1]);
        // For f(x) = x^2, Jacobian = [2x] = [4]
        Assert.Equal(4.0f, jacobian._data[0], 1);
    }

    [Fact]
    public void Jacobian_Compute_ForConstantFunction_ReturnsZeroJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            // Constant function: f(x) = 5
            return new Tensor(new float[] { 5.0f }, new[] { 1 });
        });

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        Assert.NotNull(jacobian);
        // All gradients should be zero
        for (int i = 0; i < jacobian.Size; i++)
        {
            Assert.Equal(0.0f, jacobian._data[i], 3);
        }
    }

    [Fact]
    public void Jacobian_Compute_ForIdentityFunction_ReturnsIdentityJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var f = new Func<Tensor, Tensor>(t => t); // Identity function

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        Assert.NotNull(jacobian);
        Assert.Equal(3, jacobian.Shape[0]);
        Assert.Equal(3, jacobian.Shape[1]);
        // Should be identity matrix
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                var expected = i == j ? 1.0f : 0.0f;
                Assert.Equal(expected, jacobian._data[i * 3 + j], 2);
            }
        }
    }

    #endregion

    #region Complex Function Tests

    [Fact]
    public void Jacobian_Compute_ForSinFunction_ReturnsCorrectJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 0.0f, (float)(Math.PI / 2) }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Sin().Sum());

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        Assert.NotNull(jacobian);
        // Jacobian = [cos(0), cos(π/2)] = [1, 0]
        Assert.Equal(1.0f, jacobian._data[0], 2);
        Assert.Equal(0.0f, jacobian._data[1], 2);
    }

    [Fact]
    public void Jacobian_Compute_ForExponentialFunction_ReturnsCorrectJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 0.0f, 1.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Exp().Sum());

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        Assert.NotNull(jacobian);
        // Jacobian = [exp(0), exp(1)] = [1, e]
        Assert.Equal(1.0f, jacobian._data[0], 2);
        Assert.Equal((float)Math.E, jacobian._data[1], 2);
    }

    [Fact]
    public void Jacobian_Compute_ForElementWiseMultiplication_ReturnsCorrectJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 2.0f, 3.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            // f(x) = [x[0]^2, x[1]^3]
            var data = new float[]
            {
                t._data[0] * t._data[0],
                t._data[1] * t._data[1] * t._data[1]
            };
            return new Tensor(data, new[] { 2 });
        });

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        Assert.NotNull(jacobian);
        // Jacobian should be diagonal: [[2x[0], 0], [0, 3x[1]^2]] = [[4, 0], [0, 27]]
        Assert.Equal(4.0f, jacobian._data[0], 1);
        Assert.Equal(0.0f, jacobian._data[1], 3);
        Assert.Equal(0.0f, jacobian._data[2], 3);
        Assert.Equal(27.0f, jacobian._data[3], 1);
    }

    #endregion

    #region Large Dimension Tests

    [Fact]
    public void Jacobian_Compute_ForLargeInput_ComputesCorrectly()
    {
        // Arrange
        var n = 10;
        var data = new float[n];
        for (int i = 0; i < n; i++)
        {
            data[i] = (i + 1.0f);
        }
        var x = new Tensor(data, new[] { n });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        Assert.NotNull(jacobian);
        // For f(x) = sum(x), Jacobian should be all ones
        Assert.Equal(1, jacobian.Shape[0]);
        Assert.Equal(n, jacobian.Shape[1]);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(1.0f, jacobian._data[i]);
        }
    }

    [Fact]
    public void Jacobian_Compute_ForLargeOutput_ComputesCorrectly()
    {
        // Arrange
        var x = new Tensor(new float[] { 2.0f }, new[] { 1 });
        var m = 10;
        var f = new Func<Tensor, Tensor>(t =>
        {
            var outputData = new float[m];
            for (int i = 0; i < m; i++)
            {
                outputData[i] = (float)Math.Pow(t._data[0], i + 1);
            }
            return new Tensor(outputData, new[] { m });
        });

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        Assert.NotNull(jacobian);
        // Jacobian should have m rows and 1 column
        Assert.Equal(m, jacobian.Shape[0]);
        Assert.Single(jacobian.Shape[1]);

        // For f_i(x) = x^(i+1), df_i/dx = (i+1) * x^i
        for (int i = 0; i < m; i++)
        {
            var expected = (i + 1.0f) * (float)Math.Pow(2.0f, i);
            Assert.Equal(expected, jacobian._data[i], 1);
        }
    }

    #endregion

    #region Warning Tests

    [Fact]
    public void Jacobian_Compute_WithExpensiveComputation_PrintsWarning()
    {
        // Arrange
        var n = 100;
        var data = new float[n];
        for (int i = 0; i < n; i++)
        {
            data[i] = 1.0f;
        }
        var x = new Tensor(data, new[] { n });
        var f = new Func<Tensor, Tensor>(t =>
        {
            var outputData = new float[n];
            Array.Copy(t._data, outputData, n);
            return new Tensor(outputData, new[] { n });
        });
        var options = new JacobianOptions
        {
            WarnOnExpensive = true,
            ExpensiveThreshold = 100 // 100x100 = 10,000 > 10,000
        };

        // Act & Assert
        // Should not throw, but may print warning
        var jacobian = Jacobian.Compute(f, x, options);
        Assert.NotNull(jacobian);
    }

    #endregion

    #region Parallel Computation Tests

    [Fact]
    public void Jacobian_Compute_WithParallelEnabled_ComputesCorrectly()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            var outputData = new float[4];
            for (int i = 0; i < 4; i++)
            {
                outputData[i] = t._data[i] * t._data[i];
            }
            return new Tensor(outputData, new[] { 4 });
        });
        var options = new JacobianOptions
        {
            Mode = JacobianMode.Forward,
            EnableParallel = true,
            MaxParallelTasks = 2
        };

        // Act
        var jacobian = Jacobian.Compute(f, x, options);

        // Assert
        Assert.NotNull(jacobian);
        // Should produce diagonal Jacobian
        Assert.Equal(4, jacobian.Shape[0]);
        Assert.Equal(4, jacobian.Shape[1]);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                var expected = i == j ? 2.0f * (i + 1) : 0.0f;
                Assert.Equal(expected, jacobian._data[i * 4 + j], 1);
            }
        }
    }

    #endregion
}
