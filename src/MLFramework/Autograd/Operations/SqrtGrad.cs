using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations;

/// <summary>
/// Gradient computation for square root operation.
/// For z = sqrt(x), dz/dx = 1/(2*sqrt(x)).
/// </summary>
public class SqrtGrad : IOperationGrad
{
    /// <inheritdoc/>
    public string OperationName => "Sqrt";

    /// <inheritdoc/>
    public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
    {
        if (inputs.Length < 1)
            throw new ArgumentException("Sqrt operation requires at least 1 input", nameof(inputs));

        var x = inputs[0];

        // d(sqrt(x))/dx = 1/(2*sqrt(x))
        var gradX = ComputeSqrtGradient(gradOutput, x);

        return new Tensor[] { gradX };
    }

    /// <summary>
    /// Computes the gradient for sqrt operation.
    /// </summary>
    private static Tensor ComputeSqrtGradient(Tensor gradOutput, Tensor x)
    {
        // gradOutput / (2 * sqrt(x))
        var newData = new float[x.Size];
        const float epsilon = 1e-7f;

        for (int i = 0; i < newData.Length; i++)
        {
            if (x.Data[i] < 0)
            {
                // Sqrt of negative number is undefined (return NaN)
                newData[i] = float.NaN;
            }
            else if (x.Data[i] < epsilon)
            {
                // At x = 0, gradient is infinite
                newData[i] = float.PositiveInfinity;
            }
            else
            {
                float sqrtX = (float)Math.Sqrt(x.Data[i]);
                newData[i] = gradOutput.Data[i] / (2.0f * sqrtX);
            }
        }

        return new Tensor(newData, x.Shape);
    }
}
