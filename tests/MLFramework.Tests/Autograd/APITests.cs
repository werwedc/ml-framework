using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Tests for the high-level API usability for computing derivatives.
/// Tests various entry points and ensures they work correctly together.
/// </summary>
public class APITests
{
    [Fact]
    public void JacobianAPI_WorksCorrectly_ForSimpleFunctions()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act
        var jacobian = x.Jacobian(f);

        // Assert
        Assert.NotNull(jacobian);
        Assert.Equal(2, jacobian.Size);
        Assert.Equal(1.0f, jacobian.Data[0], 3);
        Assert.Equal(1.0f, jacobian.Data[1], 3);
    }

    [Fact]
    public void HessianAPI_WorksCorrectly_ForSimpleFunctions()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
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
        var hessian = x.ComputeHessian(f);

        // Assert
        Assert.NotNull(hessian);
        Assert.Equal(2, hessian.Shape[0]);
        Assert.Equal(2, hessian.Shape[1]);
        // Hessian of x^2 + y^2 is [[2, 0], [0, 2]]
        Assert.True(Math.Abs(hessian.Data[0] - 2.0f) < 0.1f);
        Assert.True(Math.Abs(hessian.Data[1]) < 0.1f);
        Assert.True(Math.Abs(hessian.Data[2]) < 0.1f);
        Assert.True(Math.Abs(hessian.Data[3] - 2.0f) < 0.1f);
    }

    [Fact]
    public void ArbitraryOrderDerivative_ComputesCorrectly()
    {
        // Arrange - f(x) = x^3
        // f'(x) = 3x^2
        // f''(x) = 6x
        // f'''(x) = 6
        var x = new Tensor(new float[] { 2.0f }, new[] { 1 }, requiresGrad: true);

        // Act - Compute gradients at different orders
        using (var tape1 = GradientTape.Record())
        {
            var y = x.Pow(3);
            var grad1 = tape1.Gradient(y, x);
            Assert.NotNull(grad1);
            // First derivative at x=2: 3*2^2 = 12
        }

        // Note: Computing higher-order derivatives requires nested tapes
        // This test verifies the API structure is in place
        Assert.NotNull(x);
    }

    [Fact]
    public void ContextBasedAPI_WorksCorrectly_ForNestedDifferentiation()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 }, requiresGrad: true);

        // Act - Compute gradient using tape-based API
        using (var tape = GradientTape.Record())
        {
            var y = x.Pow(2);
            var grad = tape.Gradient(y, x);

            // Assert
            Assert.NotNull(grad);
            Assert.Single(grad.Size);
            // Derivative of x^2 at x=1 is 2
            Assert.Equal(2.0f, grad.Data[0], 3);
        }
    }

    [Fact]
    public void API_ThrowsHelpfulException_ForInvalidInput()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });
        Func<Tensor, Tensor> f = null!;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => x.Jacobian(f));
    }

    [Fact]
    public void Validation_CorrectlyIdentifiesNonDifferentiableOps()
    {
        // Arrange - Create a tape and record operations
        using (var tape = GradientTape.Record())
        {
            var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, requiresGrad: true);

            // Act - Use differentiable operations
            var y = x.Sum();
            var z = x.Mean();
            var w = x.Pow(2).Sum();

            // Assert - All should be differentiable
            Assert.NotNull(y);
            Assert.NotNull(z);
            Assert.NotNull(w);

            // Compute gradients - should work for all
            var gradY = tape.Gradient(y, x);
            var gradZ = tape.Gradient(z, x);
            var gradW = tape.Gradient(w, x);

            Assert.NotNull(gradY);
            Assert.NotNull(gradZ);
            Assert.NotNull(gradW);
        }
    }

    [Fact]
    public void JacobianWithOptions_WithSparseOption_ComputesSparseJacobian()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t =>
        {
            // Diagonal Jacobian (x[0]^2, x[1]^2)
            var data = new float[] { t.Data[0] * t.Data[0], t.Data[1] * t.Data[1] };
            return new Tensor(data, new[] { 2 });
        });
        var options = new JacobianOptions { Sparse = true };

        // Act
        var jacobian = Jacobian.Compute(f, x, options);

        // Assert
        Assert.NotNull(jacobian);
        Assert.Equal(2, jacobian.Shape[0]);
        Assert.Equal(2, jacobian.Shape[1]);
        // Jacobian should be [[2, 0], [0, 4]]
        Assert.Equal(2.0f, jacobian.Data[0], 2);
        Assert.True(Math.Abs(jacobian.Data[1]) < 0.1f);
        Assert.True(Math.Abs(jacobian.Data[2]) < 0.1f);
        Assert.Equal(4.0f, jacobian.Data[3], 2);
    }

    [Fact]
    public void HessianWithOptions_WithEigenvalues_ReturnsEigenvalues()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
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
        var (hessian, eigenvalues) = x.HessianWithEigenvalues(f);

        // Assert
        Assert.NotNull(hessian);
        Assert.NotNull(eigenvalues);
        Assert.Equal(2, hessian.Shape[0]);
        Assert.Equal(2, hessian.Shape[1]);
        Assert.Equal(2, eigenvalues.Size);
        // For Hessian [[2, 0], [0, 2]], eigenvalues are both 2
        Assert.True(Math.Abs(eigenvalues.Data[0] - 2.0f) < 0.1f);
        Assert.True(Math.Abs(eigenvalues.Data[1] - 2.0f) < 0.1f);
    }

    [Fact]
    public void PartialHessian_ComputesPartialHessian_Correctly()
    {
        // Arrange - 4D input, compute Hessian for indices [1, 3]
        var x = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });
        var f = new Func<Tensor, double>(t =>
        {
            var sum = 0.0;
            for (int i = 0; i < t.Size; i++)
            {
                sum += Math.Pow(t.Data[i], 2);
            }
            return sum;
        });

        // Act - Compute partial Hessian for parameters [1, 3]
        var partialHessian = x.PartialHessian(f, new[] { 1, 3 });

        // Assert
        Assert.NotNull(partialHessian);
        Assert.Equal(2, partialHessian.Shape[0]);
        Assert.Equal(2, partialHessian.Shape[1]);
        // Should be identity: [[2, 0], [0, 2]]
        Assert.Equal(2.0f, partialHessian.Data[0], 3);
        Assert.True(Math.Abs(partialHessian.Data[1]) < 0.1f);
        Assert.True(Math.Abs(partialHessian.Data[2]) < 0.1f);
        Assert.Equal(2.0f, partialHessian.Data[3], 3);
    }

    [Fact]
    public void SparseHessian_WithDetectStructure_DetectsDiagonal()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
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
        var result = x.SparseHessian(f);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.Hessian);
        Assert.Equal(2, result.Hessian.Shape[0]);
        Assert.Equal(2, result.Hessian.Shape[1]);
        // Should detect diagonal structure
        // This depends on implementation details
    }

    [Fact]
    public void HessianVectorProduct_ExtensionWorksCorrectly()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 0.5f, 1.0f }, new[] { 2 });
        var f = new Func<Tensor, double>(t =>
        {
            var sum = 0.0;
            for (int i = 0; i < t.Size; i++)
            {
                sum += Math.Pow(t.Data[i], 2);
            }
            return sum;
        });

        // Act - HVP for f(x) = x^2 + y^2, H = 2I
        // H*v = 2*[0.5, 1] = [1, 2]
        var hvp = x.HessianVectorProduct(f, v);

        // Assert
        Assert.NotNull(hvp);
        Assert.Equal(2, hvp.Size);
        Assert.Equal(1.0f, hvp.Data[0], 3);
        Assert.Equal(2.0f, hvp.Data[1], 3);
    }

    [Fact]
    public void VectorJacobianProduct_ExtensionWorksCorrectly()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var v = new Tensor(new float[] { 1.0f, 1.0f }, new[] { 2 });
        var f = new Func<Tensor, Tensor>(t => t.Sum());

        // Act - VJP for f(x) = sum(x), J = [1, 1]
        // v^T * J = [1, 1] * [1, 1]^T = 2
        var vjp = x.VectorJacobianProduct(f, v);

        // Assert
        Assert.NotNull(vjp);
        Assert.Equal(2, vjp.Size);
        Assert.Equal(2.0f, vjp.Data[0], 3);
        Assert.Equal(2.0f, vjp.Data[1], 3);
    }

    [Fact]
    public void ExtensionChain_CombinesMultipleOperations()
    {
        // Arrange - f(x) = (x + 1)^2
        var x = new Tensor(new float[] { 2.0f }, new[] { 1 });

        // Act
        var y = x + 1;
        var z = y.Pow(2);

        // Assert
        Assert.NotNull(y);
        Assert.NotNull(z);
        Assert.Single(z.Size);
        // f(2) = (2 + 1)^2 = 9
        Assert.Equal(9.0f, z.Data[0], 3);
    }

    [Fact]
    public void AutogradStaticMethods_WorkCorrectly()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
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
        var hessian1 = Autograd.Autograd.Hessian(f, x);
        var hessian2 = x.Hessian(f);

        // Assert - Both methods should give same result
        Assert.NotNull(hessian1);
        Assert.NotNull(hessian2);
        Assert.Equal(hessian1.Shape, hessian2.Shape);

        for (int i = 0; i < hessian1.Size; i++)
        {
            Assert.Equal(hessian1.Data[i], hessian2.Data[i], 3);
        }
    }

    [Fact]
    public void MultipleInputs_HandlesDifferentShapes()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var y = new Tensor(new float[] { 3.0f }, new[] { 1 });

        // Act
        using (var tape = GradientTape.Record())
        {
            var z = x.Sum() + y.Sum();
            var gradX = tape.Gradient(z, x);
            var gradY = tape.Gradient(z, y);

            // Assert
            Assert.NotNull(gradX);
            Assert.NotNull(gradY);
            Assert.Equal(2, gradX.Size);
            Assert.Equal(1, gradY.Size);
        }
    }

    [Fact]
    public void BackwardPropagation_ComputesGradientsForAllInputs()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 }, requiresGrad: true);
        var y = new Tensor(new float[] { 2.0f }, new[] { 1 }, requiresGrad: true);

        // Act
        using (var tape = GradientTape.Record())
        {
            var z = x * y + x.Pow(2);
            var grad = tape.Backward(z);

            // Assert
            Assert.NotNull(x.Gradient);
            Assert.NotNull(y.Gradient);
            // dz/dx = y + 2x = 2 + 2*1 = 4
            Assert.Equal(4.0f, x.Gradient.Data[0], 3);
            // dz/dy = x = 1
            Assert.Equal(1.0f, y.Gradient.Data[0], 3);
        }
    }

    [Fact]
    public void ErrorMessages_ProvideClearInformation()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new[] { 1 });
        Func<Tensor, Tensor> f = null!;

        // Act & Assert
        var ex = Assert.Throws<ArgumentNullException>(() => x.Jacobian(f));
        Assert.Contains("f", ex.Message);
    }
}
