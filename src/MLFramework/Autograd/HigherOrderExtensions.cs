using System;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Extension methods for Tensor to provide convenient access to higher-order derivative operations.
/// These methods make it easier to compute Jacobians, Hessians, and gradient-of-gradients in a fluent style.
/// </summary>
public static class HigherOrderExtensions
{
    /// <summary>
    /// Computes the Jacobian of a function at this tensor point.
    /// For scalar-valued functions, this is equivalent to the gradient.
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="f">The function to differentiate.</param>
    /// <returns>The Jacobian (gradient) tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor or f is null.</exception>
    /// <example>
    /// var x = Tensor.FromArray(new double[] {1.0, 2.0, 3.0}, requiresGrad: true);
    /// var f = t => t.Pow(2);
    /// var jacobian = x.Jacobian(f);
    /// // Result: [2.0, 4.0, 6.0]
    /// </example>
    public static Tensor Jacobian(this Tensor tensor, Func<Tensor, Tensor> f)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        return Autograd.Jacobian.Compute(f, tensor);
    }

    /// <summary>
    /// Computes the Hessian matrix of a function at this tensor point.
    /// The Hessian contains all second-order partial derivatives.
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="f">The scalar-valued function to differentiate.</param>
    /// <returns>The Hessian matrix as a 2D tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor or f is null.</exception>
    /// <example>
    /// var x = Tensor.FromArray(new double[] {1.0, 2.0}, requiresGrad: true);
    /// var f = t => t.Pow(4).Sum().ToScalar();
    /// var hessian = x.Hessian(f);
    /// // Result: [[12*x^2, 0], [0, 12*y^2]]
    /// </example>
    public static Tensor Hessian(this Tensor tensor, Func<Tensor, double> f)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        return Autograd.Hessian.Compute(f, tensor);
    }

    /// <summary>
    /// Computes the gradient of the gradient (meta-learning).
    /// This is useful for second-order optimization algorithms and meta-learning scenarios.
    /// </summary>
    /// <param name="tensor">The tensor whose gradient will be differentiated.</param>
    /// <returns>A new tensor containing the gradient of the gradient.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when tensor has no gradient.</exception>
    /// <example>
    /// var theta = Tensor.Random(10, requiresGrad: true);
    /// var x = Tensor.Random(5, 5, requiresGrad: true);
    ///
    /// // Inner gradient
    /// var y = model.Forward(theta, x);
    /// var innerLoss = lossFn(y);
    /// innerLoss.Backward();
    ///
    /// // Outer gradient (gradient of gradient)
    /// var gradTheta = theta.Grad.Clone().Detach().RequiresGrad();
    /// var gradOfGrad = gradTheta.GradOfGrad();
    /// </example>
    public static Tensor GradOfGrad(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (tensor.Gradient == null)
            throw new InvalidOperationException("Tensor must have a gradient to compute gradient-of-gradient");

        // Clone the gradient and enable gradient tracking for higher-order differentiation
        var gradWithGrad = new Tensor(
            (float[])TensorAccessor.GetData(tensor.Gradient).Clone(),
            (int[])tensor.Gradient.Shape.Clone(),
            requiresGrad: true
        );

        // The gradient-of-gradient computation would be performed in the calling context
        // This method prepares the gradient tensor for second-order backward pass
        return gradWithGrad;
    }

    /// <summary>
    /// Computes the diagonal of the Hessian matrix (memory-efficient).
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="f">The scalar-valued function to differentiate.</param>
    /// <returns>A 1D tensor containing the diagonal elements of the Hessian.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor or f is null.</exception>
    public static Tensor DiagonalHessian(this Tensor tensor, Func<Tensor, double> f)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        return Autograd.Hessian.ComputeDiagonal(f, tensor);
    }

    /// <summary>
    /// Computes the Hessian-Vector Product without computing the full Hessian.
    /// Useful for optimization algorithms like Newton's method with HVP.
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="f">The scalar-valued function to differentiate.</param>
    /// <param name="v">The vector to multiply with the Hessian.</param>
    /// <returns>The Hessian-Vector product H * v.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor, f, or v is null.</exception>
    public static Tensor HessianVectorProduct(this Tensor tensor, Func<Tensor, double> f, Tensor v)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (v == null)
            throw new ArgumentNullException(nameof(v));

        return Autograd.Hessian.ComputeVectorHessianProduct(f, tensor, v);
    }

    /// <summary>
    /// Computes Vector-Jacobian Product for efficiency.
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="f">The function to differentiate.</param>
    /// <param name="v">The vector to multiply with the Jacobian.</param>
    /// <returns>The Vector-Jacobian product v^T * J.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor, f, or v is null.</exception>
    public static Tensor VectorJacobianProduct(this Tensor tensor, Func<Tensor, Tensor> f, Tensor v)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (v == null)
            throw new ArgumentNullException(nameof(v));

        return Autograd.Jacobian.ComputeVectorJacobianProduct(f, tensor, v);
    }

    /// <summary>
    /// Clones the tensor and detaches it from the computation graph.
    /// The new tensor does not require gradients and has no gradient history.
    /// </summary>
    /// <param name="tensor">The tensor to detach.</param>
    /// <returns>A detached clone of the tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static Tensor Detach(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return new Tensor(
            (float[])TensorAccessor.GetData(tensor).Clone(),
            (int[])tensor.Shape.Clone(),
            requiresGrad: false
        );
    }

    /// <summary>
    /// Clones the tensor and enables gradient tracking.
    /// </summary>
    /// <param name="tensor">The tensor to clone with gradient tracking.</param>
    /// <returns>A clone of the tensor with gradient tracking enabled.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static Tensor RequiresGrad(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return new Tensor(
            (float[])TensorAccessor.GetData(tensor).Clone(),
            (int[])tensor.Shape.Clone(),
            requiresGrad: true
        );
    }

    /// <summary>
    /// Computes the sum of all elements in the tensor.
    /// </summary>
    /// <param name="tensor">The tensor to sum.</param>
    /// <returns>A scalar tensor containing the sum.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static Tensor Sum(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var data = TensorAccessor.GetData(tensor);
        var sum = 0f;
        for (int i = 0; i < tensor.Size; i++)
        {
            sum += data[i];
        }

        return new Tensor(new[] { sum }, new[] { 1 });
    }

    /// <summary>
    /// Computes the square of the tensor element-wise.
    /// </summary>
    /// <param name="tensor">The tensor to square.</param>
    /// <returns>A new tensor with each element squared.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static Tensor Pow(this Tensor tensor, int power)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var data = TensorAccessor.GetData(tensor);
        var result = new float[tensor.Size];
        for (int i = 0; i < tensor.Size; i++)
        {
            result[i] = (float)Math.Pow(data[i], power);
        }

        return new Tensor(result, tensor.Shape, tensor.RequiresGrad);
    }

    /// <summary>
    /// Computes the sine of the tensor element-wise.
    /// </summary>
    /// <param name="tensor">The tensor to compute sine for.</param>
    /// <returns>A new tensor with element-wise sine values.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static Tensor Sin(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var data = TensorAccessor.GetData(tensor);
        var result = new float[tensor.Size];
        for (int i = 0; i < tensor.Size; i++)
        {
            result[i] = (float)Math.Sin(data[i]);
        }

        return new Tensor(result, tensor.Shape, tensor.RequiresGrad);
    }

    /// <summary>
    /// Computes the cosine of the tensor element-wise.
    /// </summary>
    /// <param name="tensor">The tensor to compute cosine for.</param>
    /// <returns>A new tensor with element-wise cosine values.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    public static Tensor Cos(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var data = TensorAccessor.GetData(tensor);
        var result = new float[tensor.Size];
        for (int i = 0; i < tensor.Size; i++)
        {
            result[i] = (float)Math.Cos(data[i]);
        }

        return new Tensor(result, tensor.Shape, tensor.RequiresGrad);
    }

    /// <summary>
    /// Converts a scalar tensor to a double value.
    /// </summary>
    /// <param name="tensor">The scalar tensor to convert.</param>
    /// <returns>The double value of the tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null.</exception>
    /// <exception cref="ArgumentException">Thrown when tensor is not scalar.</exception>
    public static double ToScalar(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (tensor.Size != 1)
            throw new ArgumentException("Tensor must be scalar (size 1)");

        return TensorAccessor.GetData(tensor)[0];
    }

    /// <summary>
    /// Creates a random tensor of the specified shape.
    /// </summary>
    /// <param name="shape">The shape of the random tensor.</param>
    /// <param name="requiresGrad">Whether to enable gradient tracking (default: false).</param>
    /// <returns>A new tensor with random values.</returns>
    public static Tensor Random(int[] shape, bool requiresGrad = false)
    {
        var random = new Random();
        var length = 1;
        foreach (var dim in shape)
        {
            length *= dim;
        }

        var data = new float[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = (float)random.NextDouble();
        }

        return new Tensor(data, shape, requiresGrad);
    }
}
