using MLFramework.Autograd;
using RitterFramework.Core.Tensor;
using System;

namespace MLFramework.Autograd.Functions;

/// <summary>
/// Straight-Through Estimator (STE) for binary activation.
/// Uses a binary step function in the forward pass but passes gradients
/// through unchanged in the backward pass, enabling gradient flow through
/// binary decisions.
/// </summary>
public class STEBinary : CustomFunction
{
    private readonly double _zeroValue;

    /// <summary>
    /// Creates a new STEBinary instance.
    /// </summary>
    /// <param name="zeroValue">Value to use when input is zero (default: 0.0).</param>
    public STEBinary(double zeroValue = 0.0)
    {
        _zeroValue = zeroValue;
    }

    /// <summary>
    /// Computes the forward pass of the binary step function.
    /// </summary>
    /// <param name="inputs">Input tensors: [x].</param>
    /// <param name="ctx">Function context for saving state for backward pass.</param>
    /// <returns>Array containing the binary output tensor.</returns>
    /// <exception cref="ArgumentNullException">Thrown when input tensor is null.</exception>
    /// <exception cref="ArgumentException">Thrown when inputs array is null or empty.</exception>
    public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
    {
        if (inputs == null || inputs.Length != 1)
        {
            throw new ArgumentException("STEBinary requires exactly 1 input tensor [x]");
        }

        var x = inputs[0];
        if (x == null)
        {
            throw new ArgumentNullException(nameof(x), "Input tensor cannot be null");
        }

        // Save input for backward pass (though not strictly needed for STE)
        ctx.SaveForBackward(x);

        // Create binary tensor: 1 where x > 0, -1 where x < 0, _zeroValue where x == 0
        var resultData = new float[x.Size];

        for (int i = 0; i < x.Size; i++)
        {
            if (float.IsNaN(x.Data[i]))
            {
                // Propagate NaN
                resultData[i] = float.NaN;
            }
            else if (x.Data[i] > 0)
            {
                resultData[i] = 1.0f;
            }
            else if (x.Data[i] < 0)
            {
                resultData[i] = -1.0f;
            }
            else
            {
                // x == 0
                resultData[i] = (float)_zeroValue;
            }
        }

        var result = new Tensor(resultData, x.Shape, x.RequiresGrad, x.Dtype);
        return new[] { result };
    }

    /// <summary>
    /// Computes the backward pass - passes gradient through unchanged (Straight-Through Estimator).
    /// </summary>
    /// <param name="gradOutputs">Gradients with respect to outputs [grad_y].</param>
    /// <param name="ctx">Function context containing saved state from forward pass.</param>
    /// <returns>Array containing gradients [grad_x] (passed through unchanged).</returns>
    /// <exception cref="ArgumentNullException">Thrown when gradient tensor is null.</exception>
    /// <exception cref="ArgumentException">Thrown when gradient inputs array is null or empty.</exception>
    public override Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx)
    {
        if (gradOutputs == null)
        {
            throw new ArgumentNullException(nameof(gradOutputs), "Gradient outputs array cannot be null");
        }

        if (gradOutputs.Length != 1)
        {
            throw new ArgumentException("STEBinary Backward requires exactly 1 gradient output");
        }

        var grad_y = gradOutputs[0];
        if (grad_y == null)
        {
            throw new ArgumentNullException(nameof(grad_y), "Gradient tensor cannot be null");
        }

        // Straight-through: pass gradient through unchanged
        // This is the key STE property - gradient flows through as if the function were identity
        return new[] { grad_y };
    }
}
