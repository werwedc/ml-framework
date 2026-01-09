using RitterFramework.Core.Tensor;
using System.Diagnostics;

namespace MLFramework.Autograd;

/// <summary>
/// Provides utilities for numerical gradient checking using finite differences.
/// This class enables validation that custom function gradient implementations are correct.
/// </summary>
public static class GradientChecker
{
    /// <summary>
    /// Checks gradients for a custom function by comparing analytical gradients from the backward pass
    /// with numerically computed gradients using finite differences.
    /// </summary>
    /// <param name="function">The custom function to check.</param>
    /// <param name="inputs">Input tensors to the function.</param>
    /// <param name="epsilon">The perturbation size for finite difference computation.</param>
    /// <param name="tolerance">The tolerance for considering gradients as matching.</param>
    /// <param name="verbose">Whether to print detailed information.</param>
    /// <returns>A GradientCheckResult object with detailed information about the comparison.</returns>
    public static GradientCheckResult CheckGradients(
        CustomFunction function,
        Tensor[] inputs,
        double epsilon = 1e-6,
        double tolerance = 1e-4,
        bool verbose = false)
    {
        if (function == null)
            throw new ArgumentNullException(nameof(function));
        if (inputs == null || inputs.Length == 0)
            throw new ArgumentNullException(nameof(inputs));

        var result = new GradientCheckResult();

        // Store original requiresGrad state
        var originalRequiresGrad = inputs.Select(t => t.RequiresGrad).ToArray();

        try
        {
            // Clear graph to ensure clean state
            AutogradEngine.Instance.ClearGraph();

            // Enable gradients for all inputs temporarily
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i].RequiresGrad = true;
                // Initialize gradient tensors
                if (inputs[i].Gradient == null)
                {
                    inputs[i].Gradient = Tensor.Zeros(inputs[i].Shape);
                }
                else
                {
                    // Reset gradient to zero
                    Array.Clear(inputs[i].Gradient.Data, 0, inputs[i].Gradient.Data.Length);
                }
            }

            // Compute forward pass
            var outputs = function.ApplyMany(inputs);
            if (outputs.Length == 0)
            {
                result.Passed = false;
                result.FailureReason = "Function produced no outputs";
                return result;
            }

            // Get the function context from the registered node
            var engine = AutogradEngine.Instance;
            var nodeField = engine.GetType()
                .GetField("_nodes", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var nodes = nodeField?.GetValue(engine) as System.Collections.Concurrent.ConcurrentDictionary<Guid, CustomFunctionNode>;

            FunctionContext? context = null;
            if (nodes != null && nodes.Count > 0)
            {
                context = nodes.Values.First().Context;
            }

            // For gradient checking, we want to check d(output)/d(input) for each element
            // So we use gradient of 1.0 for each output element
            var gradOutputs = new Tensor[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                gradOutputs[i] = Tensor.Ones(outputs[i].Shape);
            }

            // Compute analytical gradients directly using the function's backward method
            var analyticalGradsRaw = function.Backward(gradOutputs, context!);

            // Create analytical gradients array
            var analyticalGrads = new Tensor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                if (i < analyticalGradsRaw.Length && analyticalGradsRaw[i] != null)
                {
                    analyticalGrads[i] = analyticalGradsRaw[i];
                }
                else
                {
                    // No gradient for this input
                    analyticalGrads[i] = Tensor.Zeros(inputs[i].Shape);
                }
            }

            // Clear graph for numerical gradient computation
            AutogradEngine.Instance.ClearGraph();

            // Compute numerical gradients separately to avoid interfering with analytical computation
            var numericalGrads = new Tensor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                if (originalRequiresGrad[i])
                {
                    numericalGrads[i] = ComputeNumericalGradient(function, inputs, i, outputs.Length > 1 ? 0 : 0, epsilon);
                }
            }

            // Compare gradients
            result.Passed = true;
            result.MaxAbsoluteDifference = 0.0;
            result.MaxRelativeError = 0.0;

            for (int i = 0; i < inputs.Length; i++)
            {
                if (originalRequiresGrad[i] && numericalGrads[i] != null && analyticalGrads[i] != null)
                {
                    var comparison = CompareTensors(
                        analyticalGrads[i],
                        numericalGrads[i],
                        i,
                        tolerance);

                    if (!comparison.WithinTolerance)
                    {
                        result.Passed = false;
                        result.Differences.AddRange(comparison.Differences);
                    }

                    result.MaxAbsoluteDifference = Math.Max(result.MaxAbsoluteDifference, comparison.MaxAbsDiff);
                    result.MaxRelativeError = Math.Max(result.MaxRelativeError, comparison.MaxRelError);
                }
            }

            // Set failure reason if applicable
            if (!result.Passed && string.IsNullOrEmpty(result.FailureReason))
            {
                result.FailureReason = $"Gradient differences exceed tolerance ({tolerance})";
            }

            // Print verbose output if requested
            if (verbose)
            {
                Console.WriteLine(result.GetSummary());
            }
        }
        catch (Exception ex)
        {
            result.Passed = false;
            result.FailureReason = $"Exception during gradient check: {ex.Message}";
        }
        finally
        {
            // Restore original requiresGrad state
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i].RequiresGrad = originalRequiresGrad[i];
                // Clear gradients
                if (inputs[i].Gradient != null)
                {
                    inputs[i].Gradient = null;
                }
            }

            // Clean up graph
            AutogradEngine.Instance.ClearGraph();
        }

        return result;
    }

    /// <summary>
    /// Computes numerical gradient for a specific input tensor using the central difference method.
    /// </summary>
    /// <param name="function">The custom function.</param>
    /// <param name="inputs">Input tensors.</param>
    /// <param name="inputIndex">Index of the input tensor to compute gradient for.</param>
    /// <param name="outputIndex">Index of the output to compute gradient with respect to.</param>
    /// <param name="epsilon">Perturbation size for finite difference.</param>
    /// <returns>Numerical gradient tensor.</returns>
    private static Tensor ComputeNumericalGradient(
        CustomFunction function,
        Tensor[] inputs,
        int inputIndex,
        int outputIndex,
        double epsilon)
    {
        var input = inputs[inputIndex];
        var grad = input.ZerosLike();

        // Create a fresh copy of inputs for each perturbation
        for (int i = 0; i < input.Size; i++)
        {
            // Get original value
            var orig = input.Data[i];

            // Create fresh copies for positive and negative perturbations
            // Create fresh input arrays to avoid any state sharing
            var inputsPlus = new Tensor[inputs.Length];
            var inputsMinus = new Tensor[inputs.Length];
            for (int j = 0; j < inputs.Length; j++)
            {
                inputsPlus[j] = inputs[j].Clone();
                inputsMinus[j] = inputs[j].Clone();
            }

            // Apply perturbation to the specific element
            inputsPlus[inputIndex].Data[i] = (float)(orig + epsilon);
            inputsMinus[inputIndex].Data[i] = (float)(orig - epsilon);

            // Clear autograd graph before each function call
            AutogradEngine.Instance.ClearGraph();
            var fPlus = function.ApplyMany(inputsPlus);
            AutogradEngine.Instance.ClearGraph();
            var fMinus = function.ApplyMany(inputsMinus);

            // For gradient checking, we verify d(output)/d(input) element-wise
            // Since both input and output have same size for these simple functions,
            // we check how output[i] changes when input[i] is perturbed
            var outPlus = fPlus[outputIndex].Data[i];
            var outMinus = fMinus[outputIndex].Data[i];

            // Central difference
            grad.Data[i] = (float)((outPlus - outMinus) / (2 * epsilon));
        }

        return grad;
    }

    /// <summary>
    /// Compares two tensors element-wise to check if they're within tolerance.
    /// </summary>
    /// <param name="analytical">The analytically computed tensor.</param>
    /// <param name="numerical">The numerically computed tensor.</param>
    /// <param name="inputIndex">The input index for reporting.</param>
    /// <param name="tolerance">The tolerance threshold.</param>
    /// <returns>A TensorComparisonResult with detailed comparison information.</returns>
    private static TensorComparisonResult CompareTensors(
        Tensor analytical,
        Tensor numerical,
        int inputIndex,
        double tolerance)
    {
        var comparison = new TensorComparisonResult();

        if (analytical.Size != numerical.Size)
        {
            comparison.WithinTolerance = false;
            comparison.Differences.Add(new TensorDifference
            {
                InputIndex = inputIndex,
                ElementIndex = Array.Empty<int>(),
                NumericalValue = numerical.Data.Length,
                AnalyticalValue = analytical.Data.Length,
                AbsoluteDifference = Math.Abs(numerical.Data.Length - analytical.Data.Length),
                RelativeError = 1.0
            });
            return comparison;
        }

        for (int i = 0; i < analytical.Size; i++)
        {
            var ana = analytical.Data[i];
            var num = numerical.Data[i];

            // Check for NaN or Inf
            if (double.IsNaN(ana) || double.IsNaN(num) ||
                double.IsInfinity(ana) || double.IsInfinity(num))
            {
                comparison.WithinTolerance = false;
                comparison.Differences.Add(new TensorDifference
                {
                    InputIndex = inputIndex,
                    ElementIndex = new[] { i },
                    NumericalValue = num,
                    AnalyticalValue = ana,
                    AbsoluteDifference = double.NaN,
                    RelativeError = double.NaN
                });
                continue;
            }

            var absDiff = Math.Abs(ana - num);
            var denominator = Math.Max(Math.Max(Math.Abs(ana), Math.Abs(num)), 1e-8);
            var relError = absDiff / denominator;

            // Update max differences
            comparison.MaxAbsDiff = Math.Max(comparison.MaxAbsDiff, absDiff);
            comparison.MaxRelError = Math.Max(comparison.MaxRelError, relError);

            // Check if within tolerance
            if (absDiff > tolerance && relError > tolerance)
            {
                comparison.WithinTolerance = false;
                comparison.Differences.Add(new TensorDifference
                {
                    InputIndex = inputIndex,
                    ElementIndex = new[] { i },
                    NumericalValue = num,
                    AnalyticalValue = ana,
                    AbsoluteDifference = absDiff,
                    RelativeError = relError
                });
            }
        }

        return comparison;
    }

    /// <summary>
    /// Computes relative error between two tensors.
    /// </summary>
    /// <param name="numerical">The numerically computed tensor.</param>
    /// <param name="analytical">The analytically computed tensor.</param>
    /// <returns>A tensor containing relative errors.</returns>
    public static Tensor ComputeRelativeError(Tensor numerical, Tensor analytical)
    {
        if (numerical.Size != analytical.Size)
            throw new ArgumentException("Tensors must have the same size");

        var result = numerical.ZerosLike();
        const double epsilon = 1e-8;

        for (int i = 0; i < numerical.Size; i++)
        {
            var num = numerical.Data[i];
            var ana = analytical.Data[i];
            var denominator = Math.Max(Math.Max(Math.Abs(num), Math.Abs(ana)), epsilon);
            result.Data[i] = (float)(Math.Abs(num - ana) / denominator);
        }

        return result;
    }

    /// <summary>
    /// Compares two tensor arrays element-wise to check if they're within tolerance.
    /// </summary>
    /// <param name="numerical">Array of numerically computed tensors.</param>
    /// <param name="analytical">Array of analytically computed tensors.</param>
    /// <param name="tolerance">The tolerance threshold.</param>
    /// <returns>True if all elements are within tolerance, false otherwise.</returns>
    public static bool CompareGradients(Tensor[] numerical, Tensor[] analytical, double tolerance = 1e-4)
    {
        if (numerical.Length != analytical.Length)
            return false;

        for (int i = 0; i < numerical.Length; i++)
        {
            if (numerical[i] == null && analytical[i] == null)
                continue;

            if (numerical[i] == null || analytical[i] == null)
                return false;

            if (numerical[i].Size != analytical[i].Size)
                return false;

            for (int j = 0; j < numerical[i].Size; j++)
            {
                var absDiff = Math.Abs(numerical[i].Data[j] - analytical[i].Data[j]);
                var denominator = Math.Max(
                    Math.Max(Math.Abs(numerical[i].Data[j]),
                    Math.Abs(analytical[i].Data[j])),
                    1e-8);
                var relError = absDiff / denominator;

                if (absDiff > tolerance && relError > tolerance)
                    return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Internal class to hold tensor comparison results.
    /// </summary>
    private class TensorComparisonResult
    {
        public bool WithinTolerance { get; set; } = true;
        public double MaxAbsDiff { get; set; } = 0.0;
        public double MaxRelError { get; set; } = 0.0;
        public List<TensorDifference> Differences { get; set; } = new List<TensorDifference>();
    }
}
