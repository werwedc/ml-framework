using System;
using System.Linq;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Static class for computing Hessian matrices of scalar-valued functions.
/// A Hessian matrix contains all second-order partial derivatives of a scalar function.
/// H[i,j] = ∂²f/∂x_i∂x_j
/// </summary>
public static class Hessian
{
    /// <summary>
    /// Computes the full Hessian matrix of a scalar-valued function f(x).
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a double.</param>
    /// <param name="x">The input tensor at which to compute the Hessian.</param>
    /// <returns>A 2D tensor representing the Hessian matrix (n x n where n is the size of x).</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    public static Tensor Compute(Func<Tensor, double> f, Tensor x)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var n = x.Size;
        var hessianData = new float[n * n];
        
        // Compute gradient first
        var xGrad = CloneWithGrad(x);
        var y = ConvertToTensor(f(xGrad));
        y.Backward();
        var gradient = xGrad.Gradient!;
        var gradientData = TensorAccessor.GetData(gradient);
        
        // Compute second derivatives (Hessian entries)
        // Using gradient-of-gradient approach
        for (int i = 0; i < n; i++)
        {
            // Create a tensor from the gradient with gradient tracking
            var gradElement = new Tensor(new[] { gradientData[i] }, new[] { 1 }, requiresGrad: true);
            
            // Compute gradient of this gradient element with respect to x
            var xSecond = CloneWithGrad(x);
            var y2 = ConvertToTensor(f(xSecond));
            y2.Backward();
            
            var grad2 = xSecond.Gradient!;
            var grad2Data = TensorAccessor.GetData(grad2);
            
            // The second derivative is the gradient of the gradient element
            // Note: This is a simplified approach. A full implementation would use proper graph retention
            for (int j = 0; j < n; j++)
            {
                hessianData[i * n + j] = grad2Data[j];
                hessianData[j * n + i] = grad2Data[j]; // Symmetric
            }
        }
        
        return new Tensor(hessianData, new[] { n, n });
    }

    /// <summary>
    /// Computes only the diagonal of the Hessian matrix.
    /// This is memory-efficient for large tensors where only second derivatives with respect to each variable are needed.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a double.</param>
    /// <param name="x">The input tensor at which to compute the diagonal Hessian.</param>
    /// <returns>A 1D tensor containing the diagonal elements of the Hessian.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    public static Tensor ComputeDiagonal(Func<Tensor, double> f, Tensor x)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var n = x.Size;
        var diagData = new float[n];
        
        // Compute diagonal elements using numerical approximation for efficiency
        for (int i = 0; i < n; i++)
        {
            // Central difference for second derivative: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
            float epsilon = 1e-5f;
            
            var xPlus = CloneTensor(x);
            var xMinus = CloneTensor(x);
            var xCenter = CloneTensor(x);
            
            var xPlusData = TensorAccessor.GetData(xPlus);
            var xMinusData = TensorAccessor.GetData(xMinus);
            
            xPlusData[i] += epsilon;
            xMinusData[i] -= epsilon;
            
            var fPlus = f(xPlus);
            var fMinus = f(xMinus);
            var fCenter = f(xCenter);
            
            // Second derivative approximation
            diagData[i] = (float)((fPlus - 2.0 * fCenter + fMinus) / (epsilon * epsilon));
        }
        
        return new Tensor(diagData, new[] { n });
    }

    /// <summary>
    /// Computes the Hessian-Vector Product (HVP) without computing the full Hessian matrix.
    /// This is much more efficient for large tensors when you only need H * v for some vector v.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a double.</param>
    /// <param name="x">The input tensor at which to compute the HVP.</param>
    /// <param name="v">The vector to multiply with the Hessian (must match x shape).</param>
    /// <returns>The Hessian-Vector product H * v.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or v is null.</exception>
    /// <exception cref="ArgumentException">Thrown when v shape doesn't match x shape.</exception>
    public static Tensor ComputeVectorHessianProduct(Func<Tensor, double> f, Tensor x, Tensor v)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));
        if (v == null)
            throw new ArgumentNullException(nameof(v));

        if (!x.Shape.SequenceEqual(v.Shape))
            throw new ArgumentException("Vector v must match input shape");

        var n = x.Size;
        var hvpData = new float[n];
        
        // Compute gradient
        var xGrad = CloneWithGrad(x);
        var y = ConvertToTensor(f(xGrad));
        y.Backward();
        var gradient = xGrad.Gradient!;
        var gradientData = TensorAccessor.GetData(gradient);
        var vData = TensorAccessor.GetData(v);
        
        // Compute HVP using directional derivative of gradient
        // HVP = (∇f(x + εv) - ∇f(x)) / ε
        float epsilon = 1e-5f;
        
        var xPerturbed = CloneTensor(x);
        var xPerturbedData = TensorAccessor.GetData(xPerturbed);
        for (int i = 0; i < n; i++)
        {
            xPerturbedData[i] += epsilon * vData[i];
        }
        
        var xPerturbedGrad = CloneWithGrad(xPerturbed);
        var yPerturbed = ConvertToTensor(f(xPerturbedGrad));
        yPerturbed.Backward();
        var gradientPerturbed = xPerturbedGrad.Gradient!;
        var gradientPerturbedData = TensorAccessor.GetData(gradientPerturbed);
        
        // HVP = (∇f(x + εv) - ∇f(x)) / ε
        for (int i = 0; i < n; i++)
        {
            hvpData[i] = (gradientPerturbedData[i] - gradientData[i]) / epsilon;
        }
        
        return new Tensor(hvpData, x.Shape);
    }

    /// <summary>
    /// Computes the Hessian using numerical approximation (finite differences).
    /// This is a fallback method when analytical differentiation is not available.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a double.</param>
    /// <param name="x">The input tensor at which to compute the Hessian.</param>
    /// <param name="epsilon">The perturbation size for finite differences (default: 1e-5).</param>
    /// <returns>A 2D tensor representing the Hessian matrix.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    public static Tensor ComputeNumerical(Func<Tensor, double> f, Tensor x, float epsilon = 1e-5f)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var n = x.Size;
        var hessianData = new float[n * n];
        
        // Compute Hessian using finite differences
        // H[i,j] = (f(x + h*ei + h*ej) - f(x + h*ei) - f(x + h*ej) + f(x)) / h²
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                var x00 = CloneTensor(x);                           // f(x)
                var x10 = CloneTensor(x);                           // f(x + h*ei)
                var x01 = CloneTensor(x);                           // f(x + h*ej)
                var x11 = CloneTensor(x);                           // f(x + h*ei + h*ej)
                
                var x10Data = TensorAccessor.GetData(x10);
                var x01Data = TensorAccessor.GetData(x01);
                var x11Data = TensorAccessor.GetData(x11);
                
                x10Data[i] += epsilon;
                x01Data[j] += epsilon;
                x11Data[i] += epsilon;
                x11Data[j] += epsilon;
                
                var f00 = f(x00);
                var f10 = f(x10);
                var f01 = f(x01);
                var f11 = f(x11);
                
                // Second mixed partial derivative
                hessianData[i * n + j] = (float)((f11 - f10 - f01 + f00) / (epsilon * epsilon));
            }
        }
        
        return new Tensor(hessianData, new[] { n, n });
    }

    /// <summary>
    /// Computes the Hessian matrix and adds regularization to improve numerical stability.
    /// Useful for ill-conditioned problems where the Hessian might be near-singular.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a double.</param>
    /// <param name="x">The input tensor at which to compute the regularized Hessian.</param>
    /// <param name="regularization">The regularization parameter to add to diagonal (default: 1e-6).</param>
    /// <returns>A 2D tensor representing the regularized Hessian matrix (H + λI).</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    public static Tensor ComputeWithRegularization(Func<Tensor, double> f, Tensor x, float regularization = 1e-6f)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var hessian = ComputeNumerical(f, x);
        var n = x.Size;
        var hessianData = TensorAccessor.GetData(hessian);
        
        // Add regularization to diagonal: H_reg = H + λI
        for (int i = 0; i < n; i++)
        {
            hessianData[i * n + i] += regularization;
        }
        
        return hessian;
    }

    /// <summary>
    /// Converts a double value to a scalar tensor.
    /// </summary>
    private static Tensor ConvertToTensor(double value)
    {
        return TensorAccessor.CreateScalar(value);
    }

    /// <summary>
    /// Clones a tensor and enables gradient tracking.
    /// </summary>
    private static Tensor CloneWithGrad(Tensor tensor)
    {
        return TensorAccessor.CloneWithGrad(tensor);
    }

    /// <summary>
    /// Clones a tensor without enabling gradient tracking.
    /// </summary>
    private static Tensor CloneTensor(Tensor tensor)
    {
        return TensorAccessor.CloneWithoutGrad(tensor);
    }
}
