using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations;

/// <summary>
/// Gradient computation for natural logarithm operation.
/// For z = log(x), dz/dx = 1/x.
/// </summary>
public class LogGrad : IOperationGrad
{
    /// <inheritdoc/>
    public string OperationName => "Log";

    /// <inheritdoc/>
    public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
    {
        if (inputs.Length < 1)
            throw new ArgumentException("Log operation requires at least 1 input", nameof(inputs));

        var x = inputs[0];

        // d(log(x))/dx = 1/x
        var gradX = ComputeLogGradient(gradOutput, x);

        return new Tensor[] { gradX };
    }

    /// <summary>
    /// Computes the gradient for log operation.
    /// </summary>
    private static Tensor ComputeLogGradient(Tensor gradOutput, Tensor x)
    {
        // 1/x * gradOutput
        var newData = new float[x.Size];
        const float epsilon = 1e-7f;

        for (int i = 0; i < newData.Length; i++)
        {
            // Avoid division by zero
            if (Math.Abs(x.Data[i]) < epsilon)
            {
                newData[i] = (x.Data[i] >= 0) ? float.PositiveInfinity : float.NaN;
            }
            else if (x.Data[i] < 0)
            {
                // Log of negative number is undefined
                newData[i] = float.NaN;
            }
            else
            {
                newData[i] = gradOutput.Data[i] / x.Data[i];
            }
        }

        return new Tensor(newData, x.Shape);
    }
}
