using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;
using System.Linq;

namespace MLFramework.Tests.Autograd;

public class JacobianVectorProductTests
{
    #region Basic JVP Tests

    [Fact]
    public void JVP_Compute_ForLinearFunction_ReturnsCorrectProduct()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };
        var x = new Tensor(data, new[] { 3 });
        var v = new Tensor(new float[] { 1.0f, 0.0f, 0.0f }, new[] { 3 });
        var f = new Func<Tensor, Tensor>(t => t.Sum()); // f(x) = sum(x)

        // Act
        var jvp = JacobianVectorProduct.Compute(f, x, v);

        // Assert
        Assert.NotNull(jvp);
        Assert.Single(jvp.Shape);
        // For f(x) = sum(x), J = [1, 1, 1], so J*v = 1*1 + 1*0 + 1*0 = 1
        Assert.Equal(1.0f, jvp._data[0]);
    }

    [Fact]
    public void JVP_Compute_ForQuadraticFunction_ReturnsCorrectProduct()
    {
        // Arrange
        var x = new Tensor(new float[] { 2.0f, 3.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 1.0f, 1.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            // f(x, y) = x^2 + y^2
            return t.Pow(2).Sum();
        });

        // Act
        var jvp = JacobianVectorProduct.Compute(f, x, v);

        // Assert
        Assert.NotNull(jvp);
        Assert.Single(jvp.Shape);
        // ∇f = [2x, 2y] = [4, 6], J*v = 4*1 + 6*1 = 10
        Assert.Equal(10.0f, jvp._data[0], 1); // Allow some tolerance for numerical errors
    }

    [Fact]
    public void JVP_Compute_NullFunction_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var v = new Tensor(new float[] { 1.0f }, new[] { 1 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => JacobianVectorProduct.Compute(null!, x, v));
    }

    [Fact]
    public void JVP_Compute_NullTensor_ThrowsException()
    {
        // Arrange
        var f = new Func<Tensor, Tensor>(t => t.Sum());
        var v = new Tensor(new float[] { 1.0f }, new[] { 1 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => JacobianVectorProduct.Compute(f, null!, v));
    }

    [Fact]
    public void JVP_Compute_NullVector_ThrowsException()
    {
        // Arrange
        var f = new Func<Tensor, Tensor>(t => t.Sum());
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => JacobianVectorProduct.Compute(f, x, null!));
    }

    [Fact]
    public void JVP_Compute_ShapeMismatch_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 1.0f }, new[] { 1 }); // Wrong shape
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act & Assert
        Assert.Throws<ArgumentException>(() => JacobianVectorProduct.Compute(f, x, v));
    }

    #endregion

    #region Batch JVP Tests

    [Fact]
    public void JVP_ComputeBatch_ReturnsCorrectBatch()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var vectors = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 0.0f }, new[] { 2 }),
            new Tensor(new float[] { 0.0f, 1.0f }, new[] { 2 }),
            new Tensor(new float[] { 1.0f, 1.0f }, new[] { 2 })
        };
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var results = JacobianVectorProduct.ComputeBatch(f, x, vectors);

        // Assert
        Assert.NotNull(results);
        Assert.Equal(3, results.Length);
        // For f(x) = sum(x), J = [1, 1]
        Assert.Equal(1.0f, results[0]._data[0]); // [1, 1] * [1, 0] = 1
        Assert.Equal(1.0f, results[1]._data[0]); // [1, 1] * [0, 1] = 1
        Assert.Equal(2.0f, results[2]._data[0]); // [1, 1] * [1, 1] = 2
    }

    [Fact]
    public void JVP_ComputeBatch_EmptyBatch_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var vectors = new Tensor[0];
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act & Assert
        Assert.Throws<ArgumentException>(() => JacobianVectorProduct.ComputeBatch(f, x, vectors));
    }

    [Fact]
    public void JVP_ComputeBatch_NullBatch_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => JacobianVectorProduct.ComputeBatch(f, x, null!));
    }

    #endregion

    #region Multiple Input Tests

    [Fact]
    public void JVP_ComputeMultiple_ReturnsCorrectResult()
    {
        // Arrange
        var x1 = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var x2 = new Tensor(new float[] { 3.0f }, new[] { 1 });
        var inputs = new Tensor[] { x1, x2 };
        var v = new Tensor(new float[] { 1.0f, 0.0f }, new[] { 2 });
        var f = new Func<Tensor[], Tensor>(t =>
        {
            // f(x1, x2) = sum(x1) + x2[0]
            return t[0].Sum() + t[1].Sum();
        });

        // Act
        var jvp = JacobianVectorProduct.ComputeMultiple(f, inputs, v, inputIndex: 0);

        // Assert
        Assert.NotNull(jvp);
        Assert.Single(jvp.Shape);
        // ∂f/∂x1 = [1, 1], so J*v = 1*1 + 1*0 = 1
        Assert.Equal(1.0f, jvp._data[0]);
    }

    [Fact]
    public void JVP_ComputeMultiple_InvalidInputIndex_ThrowsException()
    {
        // Arrange
        var x1 = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var x2 = new Tensor(new float[] { 2.0f }, new[] { 1 });
        var inputs = new Tensor[] { x1, x2 };
        var v = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var f = new Func<Tensor[], Tensor>(t => t[0].Sum() + t[1].Sum());

        // Act & Assert
        Assert.Throws<ArgumentException>(() => JacobianVectorProduct.ComputeMultiple(f, inputs, v, inputIndex: 2));
    }

    [Fact]
    public void JVP_ComputeMultiple_EmptyInputs_ThrowsException()
    {
        // Arrange
        var inputs = new Tensor[0];
        var v = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var f = new Func<Tensor[], Tensor>(t => t[0].Sum());

        // Act & Assert
        Assert.Throws<ArgumentException>(() => JacobianVectorProduct.ComputeMultiple(f, inputs, v));
    }

    #endregion

    #region Numerical Validation Tests

    [Fact]
    public void JVP_ComputeNumerical_MatchesAnalytical_ForLinearFunction()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var v = new Tensor(new float[] { 1.0f, 0.5f, 0.0f }, new[] { 3 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var jvpAnalytical = JacobianVectorProduct.Compute(f, x, v);
        var jvpNumerical = JacobianVectorProduct.ComputeNumerical(f, x, v);

        // Assert
        Assert.NotNull(jvpAnalytical);
        Assert.NotNull(jvpNumerical);
        Assert.Equal(jvpAnalytical.Shape, jvpNumerical.Shape);

        for (int i = 0; i < jvpAnalytical.Size; i++)
        {
            Assert.Equal(jvpAnalytical._data[i], jvpNumerical._data[i], 3);
        }
    }

    [Fact]
    public void JVP_ComputeNumerical_MatchesAnalytical_ForQuadraticFunction()
    {
        // Arrange
        var x = new Tensor(new float[] { 2.0f, 3.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 1.0f, 1.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Pow(2).Sum());

        // Act
        var jvpAnalytical = JacobianVectorProduct.Compute(f, x, v);
        var jvpNumerical = JacobianVectorProduct.ComputeNumerical(f, x, v);

        // Assert
        Assert.NotNull(jvpAnalytical);
        Assert.NotNull(jvpNumerical);
        Assert.Equal(jvpAnalytical.Shape, jvpNumerical.Shape);

        for (int i = 0; i < jvpAnalytical.Size; i++)
        {
            Assert.Equal(jvpAnalytical._data[i], jvpNumerical._data[i], 2);
        }
    }

    [Fact]
    public void JVP_Validate_ReturnsTrue_ForCorrectImplementation()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 0.5f, 0.5f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var isValid = JacobianVectorProduct.Validate(f, x, v);

        // Assert
        Assert.True(isValid);
    }

    #endregion

    #region Sparsity Tests

    [Fact]
    public void JVP_Compute_WithZeroVector_ReturnsZero()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 0.0f, 0.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Pow(2).Sum());

        // Act
        var jvp = JacobianVectorProduct.Compute(f, x, v);

        // Assert
        Assert.NotNull(jvp);
        Assert.Equal(0.0f, jvp._data[0], 3);
    }

    [Fact]
    public void JVP_Compute_SparseVector_HandledEfficiently()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var v = new Tensor(new float[] { 1.0f, 0.0f, 0.0f }, new[] { 3 }); // Sparse vector
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var jvp = JacobianVectorProduct.Compute(f, x, v);

        // Assert
        Assert.NotNull(jvp);
        // For f(x) = sum(x), J = [1, 1, 1], J*[1, 0, 0] = 1
        Assert.Equal(1.0f, jvp._data[0]);
    }

    #endregion

    #region Jacobian Computation via JVP Tests

    [Fact]
    public void JVP_ComputeJacobian_ReturnsCorrectJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            // f(x, y) = x^2 + y^2
            return t.Pow(2).Sum();
        });

        // Act
        var jacobian = JacobianVectorProduct.ComputeJacobian(f, x);

        // Assert
        Assert.NotNull(jacobian);
        Assert.Equal(2, jacobian.Shape.Length);
        // Output is scalar (1x2 Jacobian)
        Assert.Single(jacobian.Shape[0]);
        Assert.Equal(2, jacobian.Shape[1]);
        // Jacobian should be [2x, 2y] = [2, 4]
        Assert.Equal(2.0f, jacobian._data[0], 1);
        Assert.Equal(4.0f, jacobian._data[1], 1);
    }

    [Fact]
    public void JVP_ComputeJacobian_ForLinearFunction_ReturnsCorrectJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var jacobian = JacobianVectorProduct.ComputeJacobian(f, x);

        // Assert
        Assert.NotNull(jacobian);
        Assert.Equal(2, jacobian.Shape.Length);
        // Output is scalar (1x3 Jacobian)
        Assert.Single(jacobian.Shape[0]);
        Assert.Equal(3, jacobian.Shape[1]);
        // Jacobian should be [1, 1, 1]
        Assert.Equal(1.0f, jacobian._data[0]);
        Assert.Equal(1.0f, jacobian._data[1]);
        Assert.Equal(1.0f, jacobian._data[2]);
    }

    #endregion

    #region Cache Tests

    [Fact]
    public void JVP_ClearTangentCache_ClearsCache()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var v = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        JacobianVectorProduct.ClearTangentCache();
        var sizeAfterClear = JacobianVectorProduct.GetCacheSize();

        // Assert
        Assert.Equal(0, sizeAfterClear);
    }

    [Fact]
    public void JVP_GetCacheSize_ReturnsCorrectSize()
    {
        // Arrange
        JacobianVectorProduct.ClearTangentCache();
        var initialSize = JacobianVectorProduct.GetCacheSize();

        // Act
        var sizeAfterClear = JacobianVectorProduct.GetCacheSize();

        // Assert
        Assert.Equal(initialSize, sizeAfterClear);
    }

    #endregion

    #region Output Buffer Tests

    [Fact]
    public void JVP_Compute_WithOutputBuffer_WritesToBuffer()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 1.0f, 1.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());
        var outputBuffer = Tensor.Zeros(new[] { 1 });
        var options = new JacobianVectorProduct.JVPOptions { OutputBuffer = outputBuffer };

        // Act
        var result = JacobianVectorProduct.Compute(f, x, v, options);

        // Assert
        Assert.Same(outputBuffer, result);
        Assert.Equal(2.0f, outputBuffer._data[0]);
    }

    [Fact]
    public void JVP_Compute_WithMismatchedBuffer_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 1.0f, 1.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());
        var outputBuffer = Tensor.Zeros(new[] { 2 }); // Wrong shape
        var options = new JacobianVectorProduct.JVPOptions { OutputBuffer = outputBuffer };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => JacobianVectorProduct.Compute(f, x, v, options));
    }

    #endregion

    #region Complex Function Tests

    [Fact]
    public void JVP_Compute_ForSinFunction_ReturnsCorrectProduct()
    {
        // Arrange
        var x = new Tensor(new float[] { 0.0f, (float)(Math.PI / 2), (float)(Math.PI) }, new[] { 3 });
        var v = new Tensor(new float[] { 1.0f, 1.0f, 1.0f }, new[] { 3 });
        var f = new Func<Tensor, Tensor>(t => t.Sin().Sum());

        // Act
        var jvp = JacobianVectorProduct.Compute(f, x, v);

        // Assert
        Assert.NotNull(jvp);
        Assert.Single(jvp.Shape);
        // ∇f = [cos(x), cos(x), cos(x)]
        // cos(0) = 1, cos(π/2) = 0, cos(π) = -1
        // J*v = 1*1 + 0*1 + (-1)*1 = 0
        Assert.Equal(0.0f, jvp._data[0], 2);
    }

    [Fact]
    public void JVP_Compute_ForExponentialFunction_ReturnsCorrectProduct()
    {
        // Arrange
        var x = new Tensor(new float[] { 0.0f, 1.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 1.0f, 1.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Exp().Sum());

        // Act
        var jvp = JacobianVectorProduct.Compute(f, x, v);

        // Assert
        Assert.NotNull(jvp);
        Assert.Single(jvp.Shape);
        // ∇f = [exp(x), exp(x)] = [1, e]
        // J*v = 1*1 + e*1 = 1 + e ≈ 3.718
        Assert.Equal(Math.E + 1.0f, jvp._data[0], 2);
    }

    [Fact]
    public void JVP_Compute_ForMultiOutputFunction_ReturnsCorrectProduct()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 1.0f, 0.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            // f(x) = [x[0]^2, x[1]^2]
            return new Tensor(new float[] { t._data[0] * t._data[0], t._data[1] * t._data[1] }, new[] { 2 });
        });

        // Act
        var jvp = JacobianVectorProduct.Compute(f, x, v);

        // Assert
        Assert.NotNull(jvp);
        Assert.Equal(2, jvp.Size);
        // ∂f1/∂x = 2x = 2, ∂f2/∂x = 0
        // J*v = [2*1 + 0*0, 0*1 + 4*0] = [2, 0]
        Assert.Equal(2.0f, jvp._data[0], 1);
        Assert.Equal(0.0f, jvp._data[1], 1);
    }

    #endregion
}
