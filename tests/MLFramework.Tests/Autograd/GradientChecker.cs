using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Provides gradient checking utilities for validating autograd computations.
/// </summary>
public static class GradientChecker
{
    /// <summary>
    /// Checks if the analytical gradient matches the numerical gradient within tolerance.
    /// </summary>
    public static bool CheckGradient(Func<Tensor, Tensor> f, Tensor x, double tolerance = 1e-6)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var numericalGrad = NumericalGradient(f, x);

        // Create tensor with requiresGrad for backward pass
        var xGrad = x.Clone();
        xGrad.RequiresGrad = true;

        var y = f(xGrad);
        y.Backward();

        var analyticalGrad = xGrad.Gradient;
        if (analyticalGrad == null)
            throw new InvalidOperationException("Failed to compute analytical gradient");

        return GradientComputer.ValidateGradient(analyticalGrad, numericalGrad, tolerance);
    }

    /// <summary>
    /// Checks gradients for a function with multiple inputs.
    /// </summary>
    public static bool CheckGradients(Func<Tensor[], Tensor> f, Tensor[] inputs, double tolerance = 1e-6)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        if (inputs == null || inputs.Length == 0)
            throw new ArgumentException("Inputs must not be null or empty", nameof(inputs));

        var numericalGrads = NumericalGradients(f, inputs);

        // Enable gradients for backward pass
        var inputsGrad = inputs.Select(t =>
        {
            var tensor = t.Clone();
            tensor.RequiresGrad = true;
            return tensor;
        }).ToArray();

        var y = f(inputsGrad);
        y.Backward();

        // Check each gradient
        for (int i = 0; i < inputs.Length; i++)
        {
            var analyticalGrad = inputsGrad[i].Gradient;
            if (analyticalGrad == null)
                throw new InvalidOperationException($"Failed to compute analytical gradient for input {i}");

            if (!GradientComputer.ValidateGradient(analyticalGrad, numericalGrads[i], tolerance))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Computes the relative error between two gradients.
    /// </summary>
    public static double ComputeRelativeError(Tensor analytical, Tensor numerical)
    {
        return GradientComputer.ComputeRelativeError(analytical, numerical);
    }

    /// <summary>
    /// Computes the numerical gradient of a function.
    /// </summary>
    public static Tensor NumericalGradient(Func<Tensor, Tensor> f, Tensor x, double epsilon = 1e-6)
    {
        return GradientComputer.NumericalGradient(f, x, epsilon);
    }

    /// <summary>
    /// Checks gradient and returns detailed error information.
    /// </summary>
    public static GradientCheckResult CheckGradientDetailed(Func<Tensor, Tensor> f, Tensor x, double tolerance = 1e-6)
    {
        if (f == null)
            throw new ArgumentNullException(nameof(f));

        if (x == null)
            throw new ArgumentNullException(nameof(x));

        var numericalGrad = NumericalGradient(f, x);

        // Create tensor with requiresGrad for backward pass
        var xGrad = x.Clone();
        xGrad.RequiresGrad = true;

        var y = f(xGrad);
        y.Backward();

        var analyticalGrad = xGrad.Gradient;
        if (analyticalGrad == null)
            return new GradientCheckResult(false, double.NaN, double.NaN, null, "Failed to compute analytical gradient");

        // Compute error map
        var errorMap = Tensor.Zeros(x.Shape);
        double maxError = 0;
        double sumError = 0;

        for (int i = 0; i < x.Size; i++)
        {
            var a = analyticalGrad.Data[i];
            var n = numericalGrad.Data[i];

            double error;
            if (Math.Abs(a) < 1e-7 && Math.Abs(n) < 1e-7)
            {
                error = Math.Abs(a - n);
            }
            else
            {
                error = Math.Abs(a - n) / (Math.Abs(a) + Math.Abs(n));
            }

            errorMap.Data[i] = (float)error;
            maxError = Math.Max(maxError, error);
            sumError += error;
        }

        var meanError = sumError / x.Size;
        var passed = maxError < tolerance;

        var message = passed
            ? $"Gradient check passed (max error: {maxError:E2}, mean error: {meanError:E2})"
            : $"Gradient check failed (max error: {maxError:E2}, mean error: {meanError:E2}, tolerance: {tolerance:E2})";

        return new GradientCheckResult(passed, maxError, meanError, errorMap, message);
    }

    /// <summary>
    /// Computes numerical gradients for a function with multiple inputs.
    /// </summary>
    private static Tensor[] NumericalGradients(Func<Tensor[], Tensor> f, Tensor[] inputs, double epsilon = 1e-6)
    {
        return GradientComputer.NumericalGradients(f, inputs, epsilon);
    }
}

/// <summary>
/// Contains detailed information about a gradient check.
/// </summary>
public class GradientCheckResult
{
    public bool Passed { get; }
    public double MaxError { get; }
    public double MeanError { get; }
    public Tensor? ErrorMap { get; }
    public string Message { get; }

    public GradientCheckResult(bool passed, double maxError, double meanError, Tensor? errorMap, string message)
    {
        Passed = passed;
        MaxError = maxError;
        MeanError = meanError;
        ErrorMap = errorMap;
        Message = message;
    }
}
