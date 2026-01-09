using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd.Exceptions;

namespace MLFramework.Autograd;

/// <summary>
/// Provides validation utilities for gradient tensors to ensure correct shapes, types, and values.
/// </summary>
public static class GradientValidator
{
    #region Shape Validation

    /// <summary>
    /// Validates that gradient tensor shape matches input tensor shape.
    /// </summary>
    /// <param name="gradient">The gradient tensor to validate.</param>
    /// <param name="input">The input tensor to compare against.</param>
    /// <param name="parameterName">The name of the parameter being validated.</param>
    /// <exception cref="ArgumentNullException">Thrown when gradient or input is null.</exception>
    /// <exception cref="GradientShapeException">Thrown when shapes don't match.</exception>
    public static void ValidateGradientShape(Tensor gradient, Tensor input, string parameterName = "gradient")
    {
        if (gradient == null)
            throw new ArgumentNullException(nameof(gradient));
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        if (!AreShapesCompatible(gradient, input))
        {
            throw new GradientShapeException(
                input.Shape,
                gradient.Shape,
                parameterName);
        }
    }

    /// <summary>
    /// Checks if two tensor shapes are compatible (same or broadcastable).
    /// </summary>
    /// <param name="shape1">First tensor.</param>
    /// <param name="shape2">Second tensor.</param>
    /// <returns>True if shapes are compatible, false otherwise.</returns>
    public static bool AreShapesCompatible(Tensor shape1, Tensor shape2)
    {
        if (shape1 == null || shape2 == null)
            return false;

        int[] shape1Dims = shape1.Shape;
        int[] shape2Dims = shape2.Shape;

        // If shapes are identical, they're compatible
        if (shape1Dims.SequenceEqual(shape2Dims))
            return true;

        // Check for broadcastable shapes
        // Start from the rightmost dimension and work left
        int dim1 = shape1Dims.Length - 1;
        int dim2 = shape2Dims.Length - 1;

        while (dim1 >= 0 || dim2 >= 0)
        {
            int d1 = dim1 >= 0 ? shape1Dims[dim1] : 1;
            int d2 = dim2 >= 0 ? shape2Dims[dim2] : 1;

            // Dimensions must be equal or one of them must be 1
            if (d1 != d2 && d1 != 1 && d2 != 1)
                return false;

            dim1--;
            dim2--;
        }

        return true;
    }

    #endregion

    #region Type Validation

    /// <summary>
    /// Validates that gradient tensor dtype matches input tensor dtype.
    /// </summary>
    /// <param name="gradient">The gradient tensor to validate.</param>
    /// <param name="input">The input tensor to compare against.</param>
    /// <param name="parameterName">The name of the parameter being validated.</param>
    /// <exception cref="ArgumentNullException">Thrown when gradient or input is null.</exception>
    /// <exception cref="GradientTypeException">Thrown when dtypes don't match.</exception>
    public static void ValidateGradientType(Tensor gradient, Tensor input, string parameterName = "gradient")
    {
        if (gradient == null)
            throw new ArgumentNullException(nameof(gradient));
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        if (gradient.Dtype != input.Dtype)
        {
            throw new GradientTypeException(
                input.Dtype,
                gradient.Dtype,
                parameterName);
        }
    }

    #endregion

    #region Array Validation

    /// <summary>
    /// Validates an array of gradient tensors against their corresponding input tensors.
    /// </summary>
    /// <param name="gradients">Array of gradient tensors to validate.</param>
    /// <param name="inputs">Array of input tensors to compare against.</param>
    /// <exception cref="ArgumentNullException">Thrown when gradients or inputs is null.</exception>
    /// <exception cref="GradientShapeException">Thrown when any gradient shape doesn't match.</exception>
    /// <exception cref="GradientTypeException">Thrown when any gradient dtype doesn't match.</exception>
    public static void ValidateGradients(Tensor[] gradients, Tensor[] inputs)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        if (gradients.Length != inputs.Length)
        {
            throw new ArgumentException(
                $"Gradient count ({gradients.Length}) does not match input count ({inputs.Length})");
        }

        var errors = new List<string>();

        for (int i = 0; i < gradients.Length; i++)
        {
            var grad = gradients[i];
            var input = inputs[i];

            if (grad == null)
                continue; // Null gradients are allowed (indicates no gradient needed)

            try
            {
                ValidateGradientShape(grad, input, $"gradient[{i}]");
                ValidateGradientType(grad, input, $"gradient[{i}]");
            }
            catch (GradientShapeException ex)
            {
                errors.Add(ex.Message);
            }
            catch (GradientTypeException ex)
            {
                errors.Add(ex.Message);
            }
        }

        if (errors.Count > 0)
        {
            throw new AggregateException(
                "Multiple gradient validation failures:\n" + string.Join("\n", errors));
        }
    }

    #endregion

    #region NaN and Inf Validation

    /// <summary>
    /// Checks if gradient contains NaN values and throws an exception if found.
    /// </summary>
    /// <param name="gradient">The gradient tensor to validate.</param>
    /// <param name="parameterName">The name of the parameter being validated.</param>
    /// <exception cref="ArgumentNullException">Thrown when gradient is null.</exception>
    /// <exception cref="GradientNaNException">Thrown when NaN values are detected.</exception>
    public static void ValidateGradientHasNoNaN(Tensor gradient, string parameterName = "gradient")
    {
        if (gradient == null)
            throw new ArgumentNullException(nameof(gradient));

        for (int i = 0; i < gradient.Size; i++)
        {
            if (float.IsNaN(gradient.Data[i]))
            {
                throw new GradientNaNException(parameterName, i);
            }
        }
    }

    /// <summary>
    /// Checks if gradient contains infinite values and throws an exception if found.
    /// </summary>
    /// <param name="gradient">The gradient tensor to validate.</param>
    /// <param name="parameterName">The name of the parameter being validated.</param>
    /// <exception cref="ArgumentNullException">Thrown when gradient is null.</exception>
    /// <exception cref="GradientInfException">Thrown when infinite values are detected.</exception>
    public static void ValidateGradientHasNoInf(Tensor gradient, string parameterName = "gradient")
    {
        if (gradient == null)
            throw new ArgumentNullException(nameof(gradient));

        for (int i = 0; i < gradient.Size; i++)
        {
            if (float.IsPositiveInfinity(gradient.Data[i]))
            {
                throw new GradientInfException(parameterName, i, true);
            }
            else if (float.IsNegativeInfinity(gradient.Data[i]))
            {
                throw new GradientInfException(parameterName, i, false);
            }
        }
    }

    /// <summary>
    /// Checks if a gradient tensor contains NaN values.
    /// </summary>
    /// <param name="gradient">The gradient tensor to check.</param>
    /// <returns>True if the gradient contains NaN values, false otherwise.</returns>
    public static bool ContainsNaN(Tensor gradient)
    {
        if (gradient == null)
            throw new ArgumentNullException(nameof(gradient));

        return gradient.Data.Any(float.IsNaN);
    }

    /// <summary>
    /// Checks if a gradient tensor contains infinite values.
    /// </summary>
    /// <param name="gradient">The gradient tensor to check.</param>
    /// <returns>True if the gradient contains infinite values, false otherwise.</returns>
    public static bool ContainsInf(Tensor gradient)
    {
        if (gradient == null)
            throw new ArgumentNullException(nameof(gradient));

        return gradient.Data.Any(d => float.IsPositiveInfinity(d) || float.IsNegativeInfinity(d));
    }

    #endregion

    #region Gradient Comparison Utilities

    /// <summary>
    /// Compares two gradient tensors for approximate equality.
    /// </summary>
    /// <param name="grad1">First gradient tensor.</param>
    /// <param name="grad2">Second gradient tensor.</param>
    /// <param name="tolerance">Tolerance for floating point comparison.</param>
    /// <returns>True if gradients are approximately equal, false otherwise.</returns>
    public static bool AreGradientsEqual(Tensor grad1, Tensor grad2, double tolerance = 1e-6)
    {
        if (grad1 == null || grad2 == null)
            return false;

        if (!grad1.Shape.SequenceEqual(grad2.Shape))
            return false;

        for (int i = 0; i < grad1.Size; i++)
        {
            if (Math.Abs(grad1.Data[i] - grad2.Data[i]) > tolerance)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Computes absolute difference between two gradients.
    /// </summary>
    /// <param name="grad1">First gradient tensor.</param>
    /// <param name="grad2">Second gradient tensor.</param>
    /// <returns>New tensor with element-wise absolute differences.</returns>
    /// <exception cref="ArgumentException">Thrown when shapes don't match.</exception>
    public static Tensor GetGradientDifference(Tensor grad1, Tensor grad2)
    {
        if (grad1 == null)
            throw new ArgumentNullException(nameof(grad1));
        if (grad2 == null)
            throw new ArgumentNullException(nameof(grad2));

        if (!grad1.Shape.SequenceEqual(grad2.Shape))
            throw new ArgumentException("Gradient shapes must match for difference calculation");

        var diffData = new float[grad1.Size];
        for (int i = 0; i < grad1.Size; i++)
        {
            diffData[i] = Math.Abs(grad1.Data[i] - grad2.Data[i]);
        }

        return new Tensor(diffData, grad1.Shape, false, grad1.Dtype);
    }

    #endregion

    #region Aggregate Validation

    /// <summary>
    /// Validates gradients and returns a result object containing all validation errors.
    /// </summary>
    /// <param name="gradients">Array of gradient tensors to validate.</param>
    /// <param name="inputs">Array of input tensors to compare against.</param>
    /// <param name="checkNaN">Whether to check for NaN values.</param>
    /// <param name="checkInf">Whether to check for infinite values.</param>
    /// <returns>GradientValidationResult containing validation status and any errors.</returns>
    public static GradientValidationResult ValidateGradientsAggregate(
        Tensor[] gradients,
        Tensor[] inputs,
        bool checkNaN = true,
        bool checkInf = true)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        var result = new GradientValidationResult();

        // Validate count
        if (gradients.Length != inputs.Length)
        {
            result.Errors.Add($"Gradient count ({gradients.Length}) does not match input count ({inputs.Length})");
            result.IsValid = false;
            return result;
        }

        // Validate each gradient
        for (int i = 0; i < gradients.Length; i++)
        {
            var grad = gradients[i];
            var input = inputs[i];

            if (grad == null)
                continue; // Null gradients are allowed (indicates no gradient needed)

            // Shape check
            if (!AreShapesCompatible(grad, input))
            {
                result.Errors.Add(
                    $"Gradient [{i}] shape [{string.Join(", ", grad.Shape)}] incompatible with input shape [{string.Join(", ", input.Shape)}]");
            }

            // Type check
            if (grad.Dtype != input.Dtype)
            {
                result.Errors.Add(
                    $"Gradient [{i}] dtype {grad.Dtype} does not match input dtype {input.Dtype}");
            }

            // NaN/Inf checks
            if (checkNaN && ContainsNaN(grad))
            {
                result.Errors.Add($"Gradient [{i}] contains NaN values");
            }

            if (checkInf && ContainsInf(grad))
            {
                result.Errors.Add($"Gradient [{i}] contains Inf values");
            }
        }

        result.IsValid = result.Errors.Count == 0;
        return result;
    }

    #endregion
}

/// <summary>
/// Result of gradient validation operation containing validation status and any errors.
/// </summary>
public class GradientValidationResult
{
    /// <summary>
    /// Gets or sets whether the validation passed without errors.
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// Gets the list of validation errors.
    /// </summary>
    public List<string> Errors { get; set; } = new List<string>();

    /// <summary>
    /// Gets the list of validation warnings.
    /// </summary>
    public List<string> Warnings { get; set; } = new List<string>();
}
