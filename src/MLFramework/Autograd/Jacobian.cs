using System;
using System.Linq;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Static class for computing Jacobian matrices of tensor-valued functions.
/// A Jacobian matrix contains all first-order partial derivatives of a vector-valued function.
/// </summary>
public static class Jacobian
{
    /// <summary>
    /// Computes the Jacobian of a scalar-valued function f(x).
    /// For scalar f, the Jacobian is equivalent to the gradient ∇f(x).
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a tensor.</param>
    /// <param name="x">The input tensor at which to compute the Jacobian.</param>
    /// <returns>A tensor containing the Jacobian (gradient) values.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    /// <exception cref="ArgumentException">Thrown when x requires gradient tracking.</exception>
    public static Tensor Compute(Func<Tensor, Tensor> f, Tensor x)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));

        // Create a clone with gradient tracking enabled
        var xGrad = CloneWithGrad(x);
        
        // Forward pass
        var y = f(xGrad);
        
        // Ensure y is scalar
        if (y.Size != 1)
            throw new ArgumentException("Function output must be scalar for gradient computation");
        
        // Backward pass - gradient is stored in xGrad.Gradient
        y.Backward();
        
        // Return the gradient
        return xGrad.Gradient!;
    }

    /// <summary>
    /// Computes the Jacobian of a function with multiple input tensors.
    /// Returns a single tensor containing all partial derivatives.
    /// </summary>
    /// <param name="f">The function to differentiate, taking an array of tensors and returning a tensor.</param>
    /// <param name="inputs">Array of input tensors at which to compute the Jacobian.</param>
    /// <returns>A tensor containing the Jacobian values for all inputs.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or inputs is null.</exception>
    /// <exception cref="ArgumentException">Thrown when inputs is empty.</exception>
    public static Tensor Compute(Func<Tensor[], Tensor> f, Tensor[] inputs)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length == 0)
            throw new ArgumentException("Inputs array cannot be empty");

        // Clone all inputs with gradient tracking
        var inputsGrad = inputs.Select(CloneWithGrad).ToArray();
        
        // Forward pass
        var y = f(inputsGrad);
        
        // Ensure y is scalar
        if (y.Size != 1)
            throw new ArgumentException("Function output must be scalar for gradient computation");
        
        // Backward pass
        y.Backward();
        
        // Concatenate all gradients into a single tensor
        var totalSize = inputsGrad.Sum(t => t.Size);
        var gradData = new float[totalSize];
        var offset = 0;
        
        foreach (var input in inputsGrad)
        {
            var grad = input.Gradient!;
            var gradDataArray = TensorAccessor.GetData(grad);
            Array.Copy(gradDataArray, 0, gradData, offset, grad.Size);
            offset += grad.Size;
        }
        
        // Determine output shape (flatten all inputs)
        var outputShape = new int[] { totalSize };
        
        return new Tensor(gradData, outputShape);
    }

    /// <summary>
    /// Computes the Jacobian of a vector-valued function f(x) that returns an array of tensors.
    /// The Jacobian is a matrix where J[i,j] = ∂f_i/∂x_j.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning an array of tensors.</param>
    /// <param name="x">The input tensor at which to compute the Jacobian.</param>
    /// <returns>An array of tensors, each representing a row of the Jacobian matrix.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    /// <exception cref="ArgumentException">Thrown when f returns an empty array.</exception>
    public static Tensor[] ComputeVectorValued(Func<Tensor, Tensor[]> f, Tensor x)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));

        // Forward pass to get output dimension
        var outputs = f(x);
        if (outputs.Length == 0)
            throw new ArgumentException("Function must return at least one output tensor");

        var jacobian = new Tensor[outputs.Length];
        
        // Compute gradient for each output component
        for (int i = 0; i < outputs.Length; i++)
        {
            var xGrad = CloneWithGrad(x);
            var y = f(xGrad)[i];
            
            if (y.Size != 1)
                throw new ArgumentException($"Output {i} must be scalar for Jacobian computation");
            
            y.Backward();
            jacobian[i] = xGrad.Gradient!;
        }
        
        return jacobian;
    }

    /// <summary>
    /// Computes the Jacobian for multiple input tensors in batch.
    /// Each input tensor in the batch is differentiated independently.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a tensor.</param>
    /// <param name="inputs">Array of input tensors, each representing a batch element.</param>
    /// <returns>An array of tensors, each containing the Jacobian for one input batch.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or inputs is null.</exception>
    /// <exception cref="ArgumentException">Thrown when inputs is empty.</exception>
    public static Tensor[] ComputeBatch(Func<Tensor, Tensor> f, Tensor[] inputs)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length == 0)
            throw new ArgumentException("Inputs array cannot be empty");

        var jacobians = new Tensor[inputs.Length];
        
        for (int i = 0; i < inputs.Length; i++)
        {
            jacobians[i] = Compute(f, inputs[i]);
        }
        
        return jacobians;
    }

    /// <summary>
    /// Computes a sparse Jacobian representation using numerical approximation.
    /// This is more memory-efficient for very large tensors where the Jacobian is sparse.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a tensor.</param>
    /// <param name="x">The input tensor at which to compute the Jacobian.</param>
    /// <param name="epsilon">The perturbation size for finite differences (default: 1e-5).</param>
    /// <param name="threshold">The threshold below which gradient values are considered zero (default: 1e-6).</param>
    /// <returns>A sparse representation of the Jacobian as a dictionary mapping indices to values.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f or x is null.</exception>
    public static System.Collections.Generic.Dictionary<(int, int), float> ComputeSparse(
        Func<Tensor, Tensor> f,
        Tensor x,
        float epsilon = 1e-5f,
        float threshold = 1e-6f)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var sparseJacobian = new System.Collections.Generic.Dictionary<(int, int), float>();
        var y0 = f(x);
        var outputSize = y0.Size;
        var inputSize = x.Size;
        
        // Compute partial derivatives using central differences
        for (int j = 0; j < inputSize; j++)
        {
            // Perturb input element j
            var xPlus = CloneTensor(x);
            var xMinus = CloneTensor(x);
            
            var xPlusData = TensorAccessor.GetData(xPlus);
            var xMinusData = TensorAccessor.GetData(xMinus);
            
            // Get flat index for element j
            var flatIndex = j;
            xPlusData[flatIndex] += epsilon;
            xMinusData[flatIndex] -= epsilon;
            
            // Evaluate function at perturbed points
            var yPlus = f(xPlus);
            var yMinus = f(xMinus);
            
            var yPlusData = TensorAccessor.GetData(yPlus);
            var yMinusData = TensorAccessor.GetData(yMinus);
            
            // Compute gradient using central difference
            for (int i = 0; i < outputSize; i++)
            {
                var grad = (yPlusData[i] - yMinusData[i]) / (2.0f * epsilon);
                
                // Only store non-zero gradients (above threshold)
                if (Math.Abs(grad) > threshold)
                {
                    sparseJacobian[(i, j)] = grad;
                }
            }
        }
        
        return sparseJacobian;
    }

    /// <summary>
    /// Computes the Vector-Jacobian Product (VJP) for efficiency.
    /// This is used when you only need the product of a vector with the Jacobian.
    /// </summary>
    /// <param name="f">The function to differentiate, taking a tensor and returning a tensor.</param>
    /// <param name="x">The input tensor at which to compute the VJP.</param>
    /// <param name="v">The vector to multiply with the Jacobian (must match output size).</param>
    /// <returns>The vector-Jacobian product v^T * J.</returns>
    /// <exception cref="ArgumentNullException">Thrown when f, x, or v is null.</exception>
    /// <exception cref="ArgumentException">Thrown when v shape doesn't match output shape.</exception>
    public static Tensor ComputeVectorJacobianProduct(Func<Tensor, Tensor> f, Tensor x, Tensor v)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));
        if (x == null)
            throw new ArgumentNullException(nameof(x));
        if (v == null)
            throw new ArgumentNullException(nameof(v));

        var xGrad = CloneWithGrad(x);
        var y = f(xGrad);
        
        if (!y.Shape.SequenceEqual(v.Shape))
            throw new ArgumentException("Vector v must match output shape of function f");
        
        // Backward pass with custom gradient output (v)
        y.Backward(v);
        
        return xGrad.Gradient!;
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
