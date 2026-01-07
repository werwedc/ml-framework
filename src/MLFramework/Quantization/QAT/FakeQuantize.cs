namespace MLFramework.Quantization.QAT;

/// <summary>
/// Fake quantization operation for quantization-aware training.
/// Simulates quantization noise while preserving gradient flow through Straight-Through Estimator (STE).
/// </summary>
public class FakeQuantize
{
    private float _scale;
    private int _zeroPoint;
    private float[]? _scales; // For per-channel
    private int[]? _zeroPoints; // For per-channel
    private readonly bool _perTensor;
    private readonly int? _channelAxis;

    /// <summary>
    /// Gets the scale parameter for quantization.
    /// </summary>
    public float Scale => _scale;

    /// <summary>
    /// Gets the zero point parameter for quantization.
    /// </summary>
    public int ZeroPoint => _zeroPoint;

    /// <summary>
    /// Creates a fake quantize operation for per-tensor quantization.
    /// </summary>
    public FakeQuantize(float scale, int zeroPoint, bool perTensor = true)
    {
        _scale = scale;
        _zeroPoint = zeroPoint;
        _perTensor = perTensor;
    }

    /// <summary>
    /// Creates a fake quantize operation for per-channel quantization.
    /// </summary>
    public FakeQuantize(float[] scales, int[] zeroPoints, int channelAxis)
    {
        _scales = scales;
        _zeroPoints = zeroPoints;
        _perTensor = false;
        _channelAxis = channelAxis;
        _scale = scales[0]; // Default to first scale
        _zeroPoint = zeroPoints[0]; // Default to first zero point
    }

    /// <summary>
    /// Forward pass through fake quantization.
    /// Quantizes input and immediately dequantizes it to simulate quantization noise.
    /// </summary>
    public RitterFramework.Core.Tensor.Tensor Forward(RitterFramework.Core.Tensor.Tensor input)
    {
        var inputData = input.Data;
        var outputData = new float[inputData.Length];

        if (_perTensor)
        {
            for (int i = 0; i < inputData.Length; i++)
            {
                // Quantize: round(x / scale) + zeroPoint
                var quantized = (float)Math.Round(inputData[i] / _scale) + _zeroPoint;
                // Dequantize: (quantized - zeroPoint) * scale
                outputData[i] = (quantized - _zeroPoint) * _scale;
            }
        }
        else if (_scales != null && _zeroPoints != null)
        {
            // Per-channel quantization (simplified for 1D case)
            int channels = _scales.Length;
            int valuesPerChannel = inputData.Length / channels;

            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < valuesPerChannel; i++)
                {
                    int idx = c * valuesPerChannel + i;
                    var quantized = (float)Math.Round(inputData[idx] / _scales[c]) + _zeroPoints[c];
                    outputData[idx] = (quantized - _zeroPoints[c]) * _scales[c];
                }
            }
        }

        return RitterFramework.Core.Tensor.Tensor.FromArray(outputData, input.Dtype);
    }

    /// <summary>
    /// Backward pass through fake quantization.
    /// Uses Straight-Through Estimator (STE) - identity function for gradients.
    /// </summary>
    public RitterFramework.Core.Tensor.Tensor Backward(RitterFramework.Core.Tensor.Tensor upstreamGradient)
    {
        // STE: Identity function - pass through gradients directly
        return upstreamGradient.Clone();
    }

    /// <summary>
    /// Updates the scale and zero point parameters.
    /// </summary>
    public void UpdateScaleAndZeroPoint(float newScale, int newZeroPoint)
    {
        _scale = newScale;
        _zeroPoint = newZeroPoint;
    }
}
