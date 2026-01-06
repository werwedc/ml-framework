using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Provides utility methods for computing gradients and validating gradient computations.
/// </summary>
public static class GradientComputer
{
    /// <summary>
    /// Computes the gradient for a function using the analytical gradient computation.
    /// </summary>
    /// <param name="gradOutput">The gradient from the output.</param>
    /// <param name="input">The input tensor.</param>
    /// <param name="context">The operation context containing saved tensors and backward function.</param>
    /// <returns>The computed gradient for the input.</returns>
    public static Tensor ComputeGradient(Tensor gradOutput, Tensor input, OperationContext context)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        if (input == null)
            throw new ArgumentNullException(nameof(input));

        if (context == null)
            throw new ArgumentNullException(nameof(context));

        try
        {
            // Use the backward function from the operation context
            var gradients = context.BackwardFn(gradOutput);

            if (gradients == null || gradients.Length == 0)
                throw new InvalidOperationException("Backward function returned null or empty gradients");

            // For a single input, return the first gradient
            return gradients[0];
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to compute gradient: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Computes gradients for multiple inputs using the operation context.
    /// </summary>
    /// <param name="gradOutput">The gradient from the output.</param>
    /// <param name="inputs">The input tensors.</param>
    /// <param name="context">The operation context containing saved tensors and backward function.</param>
    /// <returns>An array of gradients, one for each input.</returns>
    public static Tensor[] ComputeGradients(Tensor gradOutput, Tensor[] inputs, OperationContext context)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        if (context == null)
            throw new ArgumentNullException(nameof(context));

        if (inputs.Length == 0)
            return Array.Empty<Tensor>();

        try
        {
            // Use the backward function from the operation context
            var gradients = context.BackwardFn(gradOutput);

            if (gradients == null)
                return Array.Empty<Tensor>();

            return gradients;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to compute gradients: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Computes the numerical gradient of a function using the finite difference method.
    /// This is useful for validating analytical gradient computations.
    /// </summary>
    /// <param name="f">The function to compute the gradient for.</param>
    /// <param name="x">The input tensor.</param>
    /// <param name="epsilon">The finite difference step size (default: 1e-6).</param>
    /// <returns>The numerical gradient tensor.</returns>
    public static Tensor NumericalGradient(Func<Tensor, Tensor> f, Tensor x, double epsilon = 1e-6)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        if (x == null)
            throw new ArgumentNullException(nameof(x));

        if (epsilon <= 0)
            throw new ArgumentException("Epsilon must be positive", nameof(epsilon));

        // Compute the function value at the original point
        var f0 = f(x);

        if (f0.Size != 1)
            throw new ArgumentException("Function must return a scalar tensor for gradient computation");

        // Create gradient tensor with same shape as input
        var grad = Tensor.Zeros(x.Shape);

        // Compute gradient for each element using central difference
        for (int i = 0; i < x.Size; i++)
        {
            // Perturb element i by +epsilon
            var xPlus = x.Clone();
            xPlus.Data[i] += (float)epsilon;
            var fPlus = f(xPlus);

            // Perturb element i by -epsilon
            var xMinus = x.Clone();
            xMinus.Data[i] -= (float)epsilon;
            var fMinus = f(xMinus);

            // Central difference formula: (f(x+e) - f(x-e)) / (2*e)
            grad.Data[i] = (float)((fPlus.Data[0] - fMinus.Data[0]) / (2 * epsilon));
        }

        return grad;
    }

    /// <summary>
    /// Computes the numerical gradient for a function with multiple inputs.
    /// </summary>
    /// <param name="f">The function to compute the gradient for.</param>
    /// <param name="inputs">The input tensors.</param>
    /// <param name="epsilon">The finite difference step size (default: 1e-6).</param>
    /// <returns>An array of numerical gradients, one for each input.</returns>
    public static Tensor[] NumericalGradients(Func<Tensor[], Tensor> f, Tensor[] inputs, double epsilon = 1e-6)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        if (inputs == null || inputs.Length == 0)
            throw new ArgumentException("Inputs must not be null or empty", nameof(inputs));

        if (epsilon <= 0)
            throw new ArgumentException("Epsilon must be positive", nameof(epsilon));

        // Compute the function value at the original point
        var f0 = f(inputs);

        if (f0.Size != 1)
            throw new ArgumentException("Function must return a scalar tensor for gradient computation");

        var gradients = new Tensor[inputs.Length];

        // Compute gradient for each input
        for (int inputIdx = 0; inputIdx < inputs.Length; inputIdx++)
        {
            var x = inputs[inputIdx];
            var grad = Tensor.Zeros(x.Shape);
            gradients[inputIdx] = grad;

            // Compute gradient for each element
            for (int i = 0; i < x.Size; i++)
            {
                // Perturb element i by +epsilon
                var inputsPlus = inputs.Select(t => t.Clone()).ToArray();
                inputsPlus[inputIdx].Data[i] += (float)epsilon;
                var fPlus = f(inputsPlus);

                // Perturb element i by -epsilon
                var inputsMinus = inputs.Select(t => t.Clone()).ToArray();
                inputsMinus[inputIdx].Data[i] -= (float)epsilon;
                var fMinus = f(inputsMinus);

                // Central difference formula
                grad.Data[i] = (float)((fPlus.Data[0] - fMinus.Data[0]) / (2 * epsilon));
            }
        }

        return gradients;
    }

    /// <summary>
    /// Computes the relative error between two gradients.
    /// Useful for validating analytical gradients against numerical gradients.
    /// </summary>
    /// <param name="analytical">The analytically computed gradient.</param>
    /// <param name="numerical">The numerically computed gradient.</param>
    /// <returns>The maximum relative error between the two gradients.</returns>
    public static double ComputeRelativeError(Tensor analytical, Tensor numerical)
    {
        if (analytical == null)
            throw new ArgumentNullException(nameof(analytical));

        if (numerical == null)
            throw new ArgumentNullException(nameof(numerical));

        if (!analytical.Shape.SequenceEqual(numerical.Shape))
            throw new ArgumentException("Gradient shapes must match");

        double maxError = 0;

        for (int i = 0; i < analytical.Size; i++)
        {
            var a = analytical.Data[i];
            var n = numerical.Data[i];

            // Compute relative error
            double error;
            if (Math.Abs(a) < 1e-7 && Math.Abs(n) < 1e-7)
            {
                // Both are near zero, use absolute error
                error = Math.Abs(a - n);
            }
            else
            {
                // Use relative error
                error = Math.Abs(a - n) / (Math.Abs(a) + Math.Abs(n));
            }

            maxError = Math.Max(maxError, error);
        }

        return maxError;
    }

    /// <summary>
    /// Computes the mean squared error between two gradients.
    /// </summary>
    /// <param name="analytical">The analytically computed gradient.</param>
    /// <param name="numerical">The numerically computed gradient.</param>
    /// <returns>The mean squared error.</returns>
    public static double ComputeMeanSquaredError(Tensor analytical, Tensor numerical)
    {
        if (analytical == null)
            throw new ArgumentNullException(nameof(analytical));

        if (numerical == null)
            throw new ArgumentNullException(nameof(numerical));

        if (!analytical.Shape.SequenceEqual(numerical.Shape))
            throw new ArgumentException("Gradient shapes must match");

        double sumSquaredError = 0;

        for (int i = 0; i < analytical.Size; i++)
        {
            var diff = analytical.Data[i] - numerical.Data[i];
            sumSquaredError += diff * diff;
        }

        return sumSquaredError / analytical.Size;
    }

    /// <summary>
    /// Validates that analytical gradients match numerical gradients within a tolerance.
    /// </summary>
    /// <param name="analytical">The analytically computed gradient.</param>
    /// <param name="numerical">The numerically computed gradient.</param>
    /// <param name="tolerance">The acceptable tolerance (default: 1e-6).</param>
    /// <returns>True if gradients match within tolerance, false otherwise.</returns>
    public static bool ValidateGradient(Tensor analytical, Tensor numerical, double tolerance = 1e-6)
    {
        var error = ComputeRelativeError(analytical, numerical);
        return error < tolerance;
    }

    /// <summary>
    /// Computes the Jacobian matrix of a vector-valued function.
    /// </summary>
    /// <param name="f">The vector-valued function.</param>
    /// <param name="x">The input tensor.</param>
    /// <param name="epsilon">The finite difference step size (default: 1e-6).</param>
    /// <returns>A matrix where each row is the gradient of an output element with respect to the input.</returns>
    public static Tensor Jacobian(Func<Tensor, Tensor> f, Tensor x, double epsilon = 1e-6)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var output = f(x);
        var jacobian = Tensor.Zeros(new int[] { output.Size, x.Size });

        for (int i = 0; i < output.Size; i++)
        {
            // Compute gradient of the i-th output element
            var outputIndex = i;

            Tensor ScalarOutputFunction(Tensor input)
            {
                var fullOutput = f(input);
                // Extract the i-th element as a scalar
                return new Tensor(new float[] { fullOutput.Data[outputIndex] }, new int[] { 1 });
            }

            var grad = NumericalGradient(ScalarOutputFunction, x, epsilon);

            // Set the i-th row of the Jacobian
            for (int j = 0; j < x.Size; j++)
            {
                jacobian.Data[i * x.Size + j] = grad.Data[j];
            }
        }

        return jacobian;
    }
}
