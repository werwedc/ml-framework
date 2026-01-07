using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;

namespace MLFramework.Quantization.QAT;

/// <summary>
/// A layer wrapper that applies fake quantization to a wrapped layer.
/// Fake quantization simulates the effects of quantization during training while maintaining gradient flow.
/// </summary>
public class FakeQuantizeLayer : ILayer
{
    private readonly FakeQuantize _inputFakeQuantize;
    private readonly FakeQuantize? _outputFakeQuantize;
    private readonly ActivationObserver? _activationObserver;
    private readonly bool _quantizeOutput;

    /// <summary>
    /// Gets the wrapped layer.
    /// </summary>
    public ILayer WrappedLayer { get; }

    /// <summary>
    /// Gets or sets the quantization parameters.
    /// </summary>
    public QuantizationParameters QuantizationParameters { get; set; }

    /// <summary>
    /// Gets or sets whether the layer is in training mode.
    /// In training mode, the observer updates statistics. In evaluation mode, statistics are frozen.
    /// </summary>
    public bool TrainingMode { get; set; } = true;

    /// <summary>
    /// Creates a fake quantize layer with per-tensor quantization.
    /// </summary>
    /// <param name="layer">The layer to wrap.</param>
    /// <param name="quantParams">The quantization parameters.</param>
    /// <param name="quantizeOutput">Whether to quantize the layer output.</param>
    public FakeQuantizeLayer(
        ILayer layer,
        QuantizationParameters quantParams,
        bool quantizeOutput = true)
    {
        WrappedLayer = layer ?? throw new ArgumentNullException(nameof(layer));
        QuantizationParameters = quantParams;
        _quantizeOutput = quantizeOutput;

        // Create fake quantize operation for input
        if (quantParams.IsPerChannel)
        {
            _inputFakeQuantize = new FakeQuantize(
                quantParams.ChannelScales ?? Array.Empty<float>(),
                quantParams.ChannelZeroPoints ?? Array.Empty<int>(),
                channelAxis: 0);
        }
        else
        {
            _inputFakeQuantize = new FakeQuantize(quantParams.Scale, quantParams.ZeroPoint, perTensor: true);
        }

        // Create fake quantize operation for output if needed
        if (quantizeOutput)
        {
            if (quantParams.IsPerChannel)
            {
                _outputFakeQuantize = new FakeQuantize(
                    quantParams.ChannelScales ?? Array.Empty<float>(),
                    quantParams.ChannelZeroPoints ?? Array.Empty<int>(),
                    channelAxis: 0);
            }
            else
            {
                _outputFakeQuantize = new FakeQuantize(quantParams.Scale, quantParams.ZeroPoint, perTensor: true);
            }
        }

        // Create activation observer for tracking statistics
        _activationObserver = new ActivationObserver(
            strategy: ObserverStrategy.MovingAverage,
            momentum: 0.9f,
            perChannel: quantParams.IsPerChannel);
    }

    /// <summary>
    /// Forward pass through the fake quantize layer.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor.</returns>
    public Tensor Forward(Tensor input)
    {
        // Step 1: Observe input statistics if in training mode
        if (TrainingMode && _activationObserver != null)
        {
            _activationObserver.Enabled = true;
            _activationObserver.Update(input);
        }
        else if (_activationObserver != null)
        {
            _activationObserver.Enabled = false;
        }

        // Step 2: Apply fake quantization to input
        var quantizedInput = _inputFakeQuantize.Forward(input);

        // Step 3: Pass through the wrapped layer
        var layerOutput = WrappedLayer.Forward(quantizedInput);

        // Step 4: Apply fake quantization to output if configured
        if (_quantizeOutput && _outputFakeQuantize != null)
        {
            layerOutput = _outputFakeQuantize.Forward(layerOutput);
        }

        // Step 5: Update quantization parameters based on observed statistics
        if (TrainingMode && _activationObserver != null)
        {
            var observedParams = _activationObserver.GetQuantizationParameters();
            if (observedParams.HasValue)
            {
                QuantizationParameters = observedParams.Value;

                // Update fake quantize operations with new parameters
                UpdateFakeQuantizeParameters(QuantizationParameters);
            }
        }

        return layerOutput;
    }

    /// <summary>
    /// Backward pass through the fake quantize layer.
    /// Uses the Straight-Through Estimator to pass gradients through fake quantization operations.
    /// </summary>
    /// <param name="upstreamGradient">The gradient from downstream operations.</param>
    /// <returns>The gradient with respect to the input.</returns>
    public Tensor Backward(Tensor upstreamGradient)
    {
        // Step 1: Pass gradient through output fake quantize (STE)
        if (_quantizeOutput && _outputFakeQuantize != null)
        {
            upstreamGradient = _outputFakeQuantize.Backward(upstreamGradient);
        }

        // Step 2: Pass gradient through wrapped layer
        var layerGradient = WrappedLayer.Backward(upstreamGradient);

        // Step 3: Pass gradient through input fake quantize (STE)
        var inputGradient = _inputFakeQuantize.Backward(layerGradient);

        return inputGradient;
    }

    /// <summary>
    /// Updates the fake quantize operations with new parameters.
    /// </summary>
    /// <param name="quantParams">The new quantization parameters.</param>
    private void UpdateFakeQuantizeParameters(QuantizationParameters quantParams)
    {
        if (quantParams.IsPerChannel && quantParams.ChannelScales != null && quantParams.ChannelZeroPoints != null)
        {
            // Update per-channel parameters
            _inputFakeQuantize.UpdateScaleAndZeroPoint(quantParams.ChannelScales[0], quantParams.ChannelZeroPoints[0]);
            _outputFakeQuantize?.UpdateScaleAndZeroPoint(quantParams.ChannelScales[0], quantParams.ChannelZeroPoints[0]);
        }
        else
        {
            // Update per-tensor parameters
            _inputFakeQuantize.UpdateScaleAndZeroPoint(quantParams.Scale, quantParams.ZeroPoint);
            _outputFakeQuantize?.UpdateScaleAndZeroPoint(quantParams.Scale, quantParams.ZeroPoint);
        }
    }

    /// <summary>
    /// Gets the observed statistics from the activation observer.
    /// </summary>
    /// <returns>The observed statistics.</returns>
    public ObserverStatistics? GetObserverStatistics()
    {
        return _activationObserver?.GetStatistics();
    }

    /// <summary>
    /// Resets the activation observer statistics.
    /// </summary>
    public void ResetObserver()
    {
        _activationObserver?.Reset();
    }
}
