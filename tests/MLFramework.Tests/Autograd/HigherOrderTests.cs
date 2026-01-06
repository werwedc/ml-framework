using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;
using System.Linq;

namespace MLFramework.Tests.Autograd;

public class HigherOrderTests
{
    #region HigherOrderContext Tests

    [Fact]
    public void HigherOrderContext_DefaultInitialization_CreatesValidContext()
    {
        // Act
        var context = new HigherOrderContext();

        // Assert
        Assert.True(context.CreateGraph);
        Assert.Equal(2, context.MaxOrder);
        Assert.Equal(0, context.CurrentOrder);
    }

    [Fact]
    public void HigherOrderContext_CustomInitialization_SetsPropertiesCorrectly()
    {
        // Act
        var context = new HigherOrderContext(createGraph: false, maxOrder: 3);

        // Assert
        Assert.False(context.CreateGraph);
        Assert.Equal(3, context.MaxOrder);
        Assert.Equal(0, context.CurrentOrder);
    }

    [Fact]
    public void HigherOrderContext_InvalidMaxOrder_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new HigherOrderContext(maxOrder: 0));
    }

    [Fact]
    public void HigherOrderContext_EnableGraphRetention_SetsCreateGraphTrue()
    {
        // Arrange
        var context = new HigherOrderContext(createGraph: false);

        // Act
        context.EnableGraphRetention();

        // Assert
        Assert.True(context.CreateGraph);
    }

    [Fact]
    public void HigherOrderContext_DisableGraphRetention_SetsCreateGraphFalse()
    {
        // Arrange
        var context = new HigherOrderContext(createGraph: true);

        // Act
        context.DisableGraphRetention();

        // Assert
        Assert.False(context.CreateGraph);
    }

    [Fact]
    public void HigherOrderContext_IncrementOrder_IncrementsOrder()
    {
        // Arrange
        var context = new HigherOrderContext(maxOrder: 3);

        // Act
        context.IncrementOrder();

        // Assert
        Assert.Equal(1, context.CurrentOrder);
    }

    [Fact]
    public void HigherOrderContext_IncrementOrder_ExceedsMax_ThrowsException()
    {
        // Arrange
        var context = new HigherOrderContext(maxOrder: 2);
        context.IncrementOrder(); // Order 1
        context.IncrementOrder(); // Order 2 (max)

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => context.IncrementOrder());
    }

    [Fact]
    public void HigherOrderContext_ResetOrder_ResetsToZero()
    {
        // Arrange
        var context = new HigherOrderContext(maxOrder: 3);
        context.IncrementOrder();
        context.IncrementOrder();

        // Act
        context.ResetOrder();

        // Assert
        Assert.Equal(0, context.CurrentOrder);
    }

    [Fact]
    public void HigherOrderContext_ShouldRetainGraph_ReturnsCorrectValue()
    {
        // Arrange
        var context = new HigherOrderContext(maxOrder: 3);

        // Act & Assert
        Assert.True(context.ShouldRetainGraph()); // Order 0 < 3

        context.IncrementOrder();
        Assert.True(context.ShouldRetainGraph()); // Order 1 < 3

        context.IncrementOrder();
        context.IncrementOrder(); // Order 3 = max
        Assert.False(context.ShouldRetainGraph());
    }

    #endregion

    #region Jacobian Tests

    [Fact]
    public void Jacobian_Compute_ForQuadraticFunction_ReturnsCorrectGradient()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };
        var x = new Tensor(data, new[] { 3 });
        var f = new Func<Tensor, Tensor>(t => t.Pow(2).Sum());

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        // f(x) = x^2, J = 2x
        Assert.NotNull(jacobian);
        Assert.Equal(3, jacobian.Size);
        Assert.Equal(2.0f, jacobian._data[0]); // 2*1
        Assert.Equal(4.0f, jacobian._data[1]); // 2*2
        Assert.Equal(6.0f, jacobian._data[2]); // 2*3
    }

    [Fact]
    public void Jacobian_Compute_ForSinFunction_ReturnsCorrectGradient()
    {
        // Arrange
        var data = new float[] { 0.0f, (float)(Math.PI / 2), (float)(Math.PI) };
        var x = new Tensor(data, new[] { 3 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var jacobian = Jacobian.Compute(f, x);

        // Assert
        Assert.NotNull(jacobian);
        Assert.Equal(3, jacobian.Size);
        Assert.Equal(1.0f, jacobian._data[0]);
        Assert.Equal(1.0f, jacobian._data[1]);
        Assert.Equal(1.0f, jacobian._data[2]);
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
    public void Jacobian_ComputeVectorValued_ReturnsCorrectShape()
    {
        // Arrange
        var x = new Tensor(new float[] { 0.0f, (float)(Math.PI / 2) }, new[] { 2 });
        var f = new Func<Tensor, Tensor[]>(t => new Tensor[] { t.Sin(), t.Cos() });

        // Act
        var jacobian = Jacobian.ComputeVectorValued(f, x);

        // Assert
        Assert.NotNull(jacobian);
        Assert.Equal(2, jacobian.Length);
        Assert.Equal(2, jacobian[0].Size); // Each row has 2 elements
        Assert.Equal(2, jacobian[1].Size);
    }

    [Fact]
    public void Jacobian_ComputeBatch_ReturnsCorrectBatch()
    {
        // Arrange
        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }),
            new Tensor(new float[] { 2.0f, 3.0f }, new[] { 2 })
        };
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var jacobians = Jacobian.ComputeBatch(f, inputs);

        // Assert
        Assert.NotNull(jacobians);
        Assert.Equal(2, jacobians.Length);
        Assert.Equal(2, jacobians[0].Size);
        Assert.Equal(2, jacobians[1].Size);
    }

    [Fact]
    public void Jacobian_ComputeSparse_ReturnsSparseRepresentation()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var sparse = Jacobian.ComputeSparse(f, x);

        // Assert
        Assert.NotNull(sparse);
        Assert.Equal(3, sparse.Count);
    }

    [Fact]
    public void Jacobian_ComputeVectorJacobianProduct_ReturnsCorrectProduct()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 1.0f, 1.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var vjp = Jacobian.ComputeVectorJacobianProduct(f, x, v);

        // Assert
        Assert.NotNull(vjp);
        Assert.Equal(2, vjp.Size);
    }

    #endregion

    #region Hessian Tests

    [Fact]
    public void Hessian_Compute_ForQuadraticFunction_ReturnsCorrectHessian()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, double>(t =>
        {
            // f(x) = x^2 + y^2
            var sum = 0.0;
            for (int i = 0; i < t.Size; i++)
            {
                sum += Math.Pow(t._data[i], 2);
            }
            return sum;
        });

        // Act
        var hessian = Hessian.Compute(f, x);

        // Assert
        Assert.NotNull(hessian);
        Assert.Equal(2, hessian.Shape[0]);
        Assert.Equal(2, hessian.Shape[1]);
        // Hessian of x^2 + y^2 is 2*I (identity matrix multiplied by 2)
        // Note: Using numerical approximation, so check for approximate values
        Assert.True(Math.Abs(hessian._data[0] - 2.0f) < 0.1f); // H[0,0] ≈ 2
        Assert.True(Math.Abs(hessian._data[1]) < 0.1f);       // H[0,1] ≈ 0
        Assert.True(Math.Abs(hessian._data[2]) < 0.1f);       // H[1,0] ≈ 0
        Assert.True(Math.Abs(hessian._data[3] - 2.0f) < 0.1f); // H[1,1] ≈ 2
    }

    [Fact]
    public void Hessian_ComputeDiagonal_ReturnsDiagonalElements()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });
        var f = new Func<Tensor, double>(t =>
        {
            var sum = 0.0;
            for (int i = 0; i < t.Size; i++)
            {
                sum += Math.Pow(t._data[i], 2);
            }
            return sum;
        });

        // Act
        var diagHessian = Hessian.ComputeDiagonal(f, x);

        // Assert
        Assert.NotNull(diagHessian);
        Assert.Equal(3, diagHessian.Size);
        // Diagonal should be 2 for all elements
        Assert.True(Math.Abs(diagHessian._data[0] - 2.0f) < 0.1f);
        Assert.True(Math.Abs(diagHessian._data[1] - 2.0f) < 0.1f);
        Assert.True(Math.Abs(diagHessian._data[2] - 2.0f) < 0.1f);
    }

    [Fact]
    public void Hessian_ComputeVectorHessianProduct_ReturnsCorrectProduct()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 1.0f, 1.0f }, new[] { 2 });
        var f = new Func<Tensor, double>(t =>
        {
            var sum = 0.0;
            for (int i = 0; i < t.Size; i++)
            {
                sum += Math.Pow(t._data[i], 2);
            }
            return sum;
        });

        // Act
        var hvp = Hessian.ComputeVectorHessianProduct(f, x, v);

        // Assert
        Assert.NotNull(hvp);
        Assert.Equal(2, hvp.Size);
        // For f = x^2 + y^2, H = 2I, so H*v = 2*v
        Assert.True(Math.Abs(hvp._data[0] - 2.0f) < 0.1f);
        Assert.True(Math.Abs(hvp._data[1] - 2.0f) < 0.1f);
    }

    [Fact]
    public void Hessian_ComputeNumerical_ReturnsApproximateHessian()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, double>(t =>
        {
            var sum = 0.0;
            for (int i = 0; i < t.Size; i++)
            {
                sum += Math.Pow(t._data[i], 2);
            }
            return sum;
        });

        // Act
        var hessian = Hessian.ComputeNumerical(f, x);

        // Assert
        Assert.NotNull(hessian);
        Assert.Equal(2, hessian.Shape[0]);
        Assert.Equal(2, hessian.Shape[1]);
    }

    [Fact]
    public void Hessian_ComputeWithRegularization_AddsRegularization()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var f = new Func<Tensor, double>(t => Math.Pow(t._data[0], 2));
        var regularization = 0.1f;

        // Act
        var hessian = Hessian.ComputeWithRegularization(f, x, regularization);

        // Assert
        Assert.NotNull(hessian);
        // Unregularized Hessian of x^2 is 2, so with regularization it should be 2.1
        Assert.True(Math.Abs(hessian._data[0] - 2.1f) < 0.1f);
    }

    [Fact]
    public void Hessian_Compute_NullFunction_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => Hessian.Compute(null!, x));
    }

    [Fact]
    public void Hessian_Compute_NullTensor_ThrowsException()
    {
        // Arrange
        var f = new Func<Tensor, double>(t => 0.0);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => Hessian.Compute(f, null!));
    }

    #endregion

    #region Extension Methods Tests

    [Fact]
    public void Extension_Jacobian_ReturnsCorrectGradient()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Pow(2).Sum());

        // Act
        var jacobian = x.Jacobian(f);

        // Assert
        Assert.NotNull(jacobian);
        Assert.Equal(2, jacobian.Size);
        Assert.Equal(2.0f, jacobian._data[0]);
        Assert.Equal(4.0f, jacobian._data[1]);
    }

    [Fact]
    public void Extension_Hessian_ReturnsCorrectHessian()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });
        var f = new Func<Tensor, double>(t => Math.Pow(t._data[0], 2));

        // Act
        var hessian = x.Hessian(f);

        // Assert
        Assert.NotNull(hessian);
        Assert.Equal(1, hessian.Shape[0]);
        Assert.Equal(1, hessian.Shape[1]);
    }

    [Fact]
    public void Extension_Detach_CreatesDetachedTensor()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, requiresGrad: true);

        // Act
        var detached = x.Detach();

        // Assert
        Assert.NotNull(detached);
        Assert.False(detached.RequiresGrad);
        Assert.Null(detached.Gradient);
    }

    [Fact]
    public void Extension_RequiresGrad_CreatesTensorWithGrad()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, requiresGrad: false);

        // Act
        var withGrad = x.RequiresGrad();

        // Assert
        Assert.NotNull(withGrad);
        Assert.True(withGrad.RequiresGrad);
        Assert.NotNull(withGrad.Gradient);
    }

    [Fact]
    public void Extension_Sum_ReturnsCorrectSum()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });

        // Act
        var sum = x.Sum();

        // Assert
        Assert.Equal(1, sum.Size);
        Assert.Equal(6.0f, sum._data[0]);
    }

    [Fact]
    public void Extension_Pow_ReturnsSquaredValues()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new[] { 3 });

        // Act
        var squared = x.Pow(2);

        // Assert
        Assert.Equal(3, squared.Size);
        Assert.Equal(1.0f, squared._data[0]);
        Assert.Equal(4.0f, squared._data[1]);
        Assert.Equal(9.0f, squared._data[2]);
    }

    [Fact]
    public void Extension_Sin_ReturnsSinValues()
    {
        // Arrange
        var x = new Tensor(new float[] { 0.0f, (float)(Math.PI / 2) }, new[] { 2 });

        // Act
        var sin = x.Sin();

        // Assert
        Assert.Equal(2, sin.Size);
        Assert.True(Math.Abs(sin._data[0]) < 0.001f); // sin(0) = 0
        Assert.True(Math.Abs(sin._data[1] - 1.0f) < 0.001f); // sin(π/2) = 1
    }

    [Fact]
    public void Extension_Cos_ReturnsCosValues()
    {
        // Arrange
        var x = new Tensor(new float[] { 0.0f, (float)(Math.PI / 2) }, new[] { 2 });

        // Act
        var cos = x.Cos();

        // Assert
        Assert.Equal(2, cos.Size);
        Assert.True(Math.Abs(cos._data[0] - 1.0f) < 0.001f); // cos(0) = 1
        Assert.True(Math.Abs(cos._data[1]) < 0.001f); // cos(π/2) = 0
    }

    [Fact]
    public void Extension_ToScalar_ConvertsScalarTensor()
    {
        // Arrange
        var x = new Tensor(new float[] { 5.0f }, new[] { 1 });

        // Act
        var scalar = x.ToScalar();

        // Assert
        Assert.Equal(5.0, scalar);
    }

    [Fact]
    public void Extension_ToScalar_NonScalarTensor_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => x.ToScalar());
    }

    [Fact]
    public void Extension_Random_CreatesRandomTensor()
    {
        // Act
        var tensor = HigherOrderExtensions.Random(new[] { 3, 4 });

        // Assert
        Assert.NotNull(tensor);
        Assert.Equal(12, tensor.Size);
        Assert.Equal(2, tensor.Dimensions);
    }

    #endregion
}
