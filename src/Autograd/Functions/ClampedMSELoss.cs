using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;

namespace MLFramework.Autograd.Functions;

/// <summary>
/// Computes Mean Squared Error loss with clamped differences.
/// This prevents extreme gradient values when predictions are far from targets,
/// providing robustness against outliers.
/// </summary>
public class ClampedMSELoss : CustomFunction
{
    private readonly float _clampMin;
    private readonly float _clampMax;
    private readonly string _reduction;

    /// <summary>
    /// Creates a new ClampedMSELoss instance.
    /// </summary>
    /// <param name="clampMin">Minimum clamp value (e.g., -1.0).</param>
    /// <param name="clampMax">Maximum clamp value (e.g., 1.0).</param>
    /// <param name="reduction">Reduction mode: 'mean', 'sum', or 'none' (default: 'mean').</param>
    /// <exception cref="ArgumentException">Thrown when clampMin >= clampMax.</exception>
    /// <exception cref="ArgumentException">Thrown when reduction is not valid.</exception>
    public ClampedMSELoss(double clampMin, double clampMax, string reduction = "mean")
    {
        if (clampMin >= clampMax)
            throw new ArgumentException($"clampMin ({clampMin}) must be less than clampMax ({clampMax})");

        var reductionLower = reduction.ToLower();
        if (reductionLower != "mean" && reductionLower != "sum" && reductionLower != "none")
            throw new ArgumentException($"reduction must be 'mean', 'sum', or 'none', got '{reduction}'");

        _clampMin = (float)clampMin;
        _clampMax = (float)clampMax;
        _reduction = reductionLower;
    }

    /// <summary>
    /// Computes the forward pass of the clamped MSE loss.
    /// </summary>
    /// <param name="inputs">Input tensors: [predictions, targets].</param>
    /// <param name="ctx">Function context for saving state for backward pass.</param>
    /// <returns>Array containing the loss tensor (scalar or per-element depending on reduction).</returns>
    /// <exception cref="ArgumentNullException">Thrown when predictions or targets are null.</exception>
    /// <exception cref="ArgumentException">Thrown when predictions and targets have different shapes.</exception>
    public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
    {
        if (inputs == null || inputs.Length != 2)
            throw new ArgumentException("ClampedMSELoss requires exactly 2 input tensors [predictions, targets]");

        var predictions = inputs[0];
        var targets = inputs[1];

        if (predictions == null)
            throw new ArgumentNullException(nameof(predictions));
        if (targets == null)
            throw new ArgumentNullException(nameof(targets));

        if (!predictions.HasSameShape(targets))
            throw new ArgumentException(
                $"Predictions and targets must have the same shape. " +
                $"Got predictions: {predictions.GetShapeString()}, targets: {targets.GetShapeString()}");

        // Compute difference and clamp it
        var diff = predictions.Subtract(targets);
        var clampedDiff = diff.Clamp(_clampMin, _clampMax);

        // Compute squared differences based on reduction mode
        Tensor loss;
        switch (_reduction)
        {
            case "none":
                // For 'none' reduction, return per-element squared losses
                loss = clampedDiff.Multiply(clampedDiff);
                break;
            case "sum":
                // For 'sum' reduction, sum all squared differences
                var squared = clampedDiff.Multiply(clampedDiff);
                loss = squared.Sum();
                break;
            case "mean":
            default:
                // For 'mean' reduction, average all squared differences
                var squaredMean = clampedDiff.Multiply(clampedDiff);
                loss = squaredMean.Mean();
                break;
        }

        // Save tensors for backward pass
        ctx.SaveForBackward(predictions, targets, diff);

        return new[] { loss };
    }

    /// <summary>
    /// Computes the backward pass of the clamped MSE loss.
    /// </summary>
    /// <param name="gradOutputs">Gradients with respect to the loss.</param>
    /// <param name="ctx">Function context containing saved state from forward pass.</param>
    /// <returns>Array containing gradients [grad_preds, grad_targets].</returns>
    public override Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx)
    {
        if (gradOutputs == null || gradOutputs.Length != 1)
            throw new ArgumentException("ClampedMSELoss Backward requires exactly 1 gradient output");

        var gradLoss = gradOutputs[0];
        var predictions = ctx.GetSavedTensor(0);
        var targets = ctx.GetSavedTensor(1);
        var diff = ctx.GetSavedTensor(2);

        // Compute clamped difference and mask
        var clampedDiff = diff.Clamp(_clampMin, _clampMax);
        var lowerBoundMask = diff.GreaterThanOrEqual(_clampMin);
        var upperBoundMask = diff.LessThanOrEqual(_clampMax);
        var mask = lowerBoundMask.And(upperBoundMask);

        // Compute gradient: 2 * clamped_diff * mask
        var gradUnclamped = clampedDiff.MultiplyScalar(2.0f);
        var gradPredsUnmasked = gradUnclamped.Multiply(mask);

        // Multiply by upstream gradient from loss
        Tensor gradPreds;
        Tensor gradTargets;

        if (gradLoss.Size == 1 && _reduction != "none")
        {
            // Loss is scalar (for sum/mean reduction), multiply by scalar gradient
            var scalarGrad = gradLoss.Data[0];
            var gradPredsData = new float[gradPredsUnmasked.Size];
            var gradTargetsData = new float[gradPredsUnmasked.Size];
            for (int i = 0; i < gradPredsUnmasked.Size; i++)
            {
                gradPredsData[i] = gradPredsUnmasked.Data[i] * scalarGrad;
                gradTargetsData[i] = -gradPredsUnmasked.Data[i] * scalarGrad;
            }
            gradPreds = new Tensor(gradPredsData, gradPredsUnmasked.Shape, predictions.RequiresGrad);
            gradTargets = new Tensor(gradTargetsData, gradPredsUnmasked.Shape, targets.RequiresGrad);
        }
        else
        {
            // For 'none' reduction, element-wise multiply
            var gradPredsData = new float[gradPredsUnmasked.Size];
            var gradTargetsData = new float[gradPredsUnmasked.Size];
            for (int i = 0; i < gradPredsUnmasked.Size; i++)
            {
                gradPredsData[i] = gradPredsUnmasked.Data[i] * gradLoss.Data[i];
                gradTargetsData[i] = -gradPredsData[i];
            }
            gradPreds = new Tensor(gradPredsData, gradPredsUnmasked.Shape, predictions.RequiresGrad);
            gradTargets = new Tensor(gradTargetsData, gradPredsUnmasked.Shape, targets.RequiresGrad);
        }

        // Apply reduction scaling for 'mean' mode
        if (_reduction == "mean")
        {
            var numElements = (float)predictions.NumberOfElements();
            var scaledPredsData = new float[gradPreds.Size];
            var scaledTargetsData = new float[gradTargets.Size];
            for (int i = 0; i < gradPreds.Size; i++)
            {
                scaledPredsData[i] = gradPreds.Data[i] / numElements;
                scaledTargetsData[i] = gradTargets.Data[i] / numElements;
            }
            gradPreds = new Tensor(scaledPredsData, gradPreds.Shape, predictions.RequiresGrad);
            gradTargets = new Tensor(scaledTargetsData, gradTargets.Shape, targets.RequiresGrad);
        }

        return new[] { gradPreds, gradTargets };
    }
}
