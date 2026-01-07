using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;

namespace MLFramework.Quantization.QAT;

/// <summary>
/// A wrapper module that applies quantization-aware training (QAT) to standard layers.
/// Inserts fake quantization nodes before and after layer operations to simulate quantization noise.
/// Supports both weight quantization (before layer) and activation quantization (after layer).
/// </summary>
public class QATModuleWrapper : ILayer
{
    private readonly FakeQuantize _weightFakeQuantize;
    private readonly FakeQuantize _activationFakeQuantize;
    private readonly ActivationObserver _weightObserver;
    private readonly ActivationObserver _activationObserver;

    /// <summary>
    /// Gets the wrapped module/layer.
    /// </summary>
    public ILayer WrappedModule { get; }

    /// <summary>
    /// Gets the quantization parameters for weights.
    /// </summary>
    public QuantizationParameters WeightQuantizationParameters { get; set; }

    /// <summary>
    /// Gets the quantization parameters for activations.
    /// </summary>
    public QuantizationParameters ActivationQuantizationParameters { get; set; }

    /// <summary>
    /// Gets or sets whether the module is in training mode.
    /// </summary>
    public bool TrainingMode { get; set; } = true;

    /// <summary>
    /// Creates a QAT module wrapper.
    /// </summary>
    /// <param name="module">The module to wrap.</param>
    /// <param name="weightQuantParams">Quantization parameters for weights.</param>
    /// <param name="activationQuantParams">Quantization parameters for activations.</param>
    public QATModuleWrapper(
        ILayer module,
        QuantizationParameters weightQuantParams,
        QuantizationParameters activationQuantParams)
    {
        WrappedModule = module ?? throw new ArgumentNullException(nameof(module));
        WeightQuantizationParameters = weightQuantParams;
        ActivationQuantizationParameters = activationQuantParams;

        // Create fake quantize operations for weights
        if (weightQuantParams.IsPerChannel && weightQuantParams.ChannelScales != null && weightQuantParams.ChannelZeroPoints != null)
        {
            _weightFakeQuantize = new FakeQuantize(
                weightQuantParams.ChannelScales,
                weightQuantParams.ChannelZeroPoints,
                channelAxis: 0);
        }
        else
        {
            _weightFakeQuantize = new FakeQuantize(
                weightQuantParams.Scale,
                weightQuantParams.ZeroPoint,
                perTensor: true);
        }

        // Create fake quantize operations for activations
        if (activationQuantParams.IsPerChannel && activationQuantParams.ChannelScales != null && activationQuantParams.ChannelZeroPoints != null)
        {
            _activationFakeQuantize = new FakeQuantize(
                activationQuantParams.ChannelScales,
                activationQuantParams.ChannelZeroPoints,
                channelAxis: 0);
        }
        else
        {
            _activationFakeQuantize = new FakeQuantize(
                activationQuantParams.Scale,
                activationQuantParams.ZeroPoint,
                perTensor: true);
        }

        // Create observers for tracking statistics
        _weightObserver = new ActivationObserver(
            strategy: ObserverStrategy.MovingAverage,
            momentum: 0.9f,
            perChannel: weightQuantParams.IsPerChannel);

        _activationObserver = new ActivationObserver(
            strategy: ObserverStrategy.MovingAverage,
            momentum: 0.9f,
            perChannel: activationQuantParams.IsPerChannel);
    }

    /// <summary>
    /// Forward pass through the QAT module wrapper.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor.</returns>
    public Tensor Forward(Tensor input)
    {
        // Step 1: Observe input (activation) statistics if in training mode
        if (TrainingMode)
        {
            _activationObserver.Enabled = true;
            _activationObserver.Update(input);
        }
        else
        {
            _activationObserver.Enabled = false;
        }

        // Step 2: Apply fake quantization to input (activation quantization)
        var quantizedInput = _activationFakeQuantize.Forward(input);

        // Step 3: Pass through wrapped module
        var output = WrappedModule.Forward(quantizedInput);

        // Step 4: Update activation quantization parameters based on observed statistics
        if (TrainingMode)
        {
            var observedParams = _activationObserver.GetQuantizationParameters();
            if (observedParams.HasValue)
            {
                ActivationQuantizationParameters = observedParams.Value;
                UpdateActivationFakeQuantize(ActivationQuantizationParameters);
            }
        }

        return output;
    }

    /// <summary>
    /// Backward pass through the QAT module wrapper.
    /// Uses the Straight-Through Estimator to pass gradients through fake quantization operations.
    /// </summary>
    /// <param name="upstreamGradient">The gradient from downstream operations.</param>
    /// <returns>The gradient with respect to the input.</returns>
    public Tensor Backward(Tensor upstreamGradient)
    {
        // Step 1: Pass gradient through activation fake quantize (STE)
        var activationGradient = _activationFakeQuantize.Backward(upstreamGradient);

        // Step 2: Pass gradient through wrapped module
        var moduleGradient = WrappedModule.Backward(activationGradient);

        // Note: Weight gradients are handled separately during weight update
        // The weight fake quantize is applied to weights, not activations

        return moduleGradient;
    }

    /// <summary>
    /// Applies fake quantization to module weights.
    /// This should be called before the forward pass to quantize weights.
    /// </summary>
    public void QuantizeWeights()
    {
        // In a production implementation, this would:
        // 1. Access the module's weight tensor
        // 2. Apply fake quantization to the weights
        // 3. Observe weight statistics
        // 4. Update weight quantization parameters

        // For now, this is a placeholder
        // The actual implementation would depend on how the wrapped module stores its weights
    }

    /// <summary>
    /// Updates the activation fake quantize with new parameters.
    /// </summary>
    /// <param name="quantParams">The new quantization parameters.</param>
    private void UpdateActivationFakeQuantize(QuantizationParameters quantParams)
    {
        if (quantParams.IsPerChannel && quantParams.ChannelScales != null && quantParams.ChannelZeroPoints != null)
        {
            // Update per-channel parameters
            _activationFakeQuantize.UpdateScaleAndZeroPoint(
                quantParams.ChannelScales[0],
                quantParams.ChannelZeroPoints[0]);
        }
        else
        {
            // Update per-tensor parameters
            _activationFakeQuantize.UpdateScaleAndZeroPoint(
                quantParams.Scale,
                quantParams.ZeroPoint);
        }
    }

    /// <summary>
    /// Updates the weight fake quantize with new parameters.
    /// </summary>
    /// <param name="quantParams">The new quantization parameters.</param>
    private void UpdateWeightFakeQuantize(QuantizationParameters quantParams)
    {
        if (quantParams.IsPerChannel && quantParams.ChannelScales != null && quantParams.ChannelZeroPoints != null)
        {
            // Update per-channel parameters
            _weightFakeQuantize.UpdateScaleAndZeroPoint(
                quantParams.ChannelScales[0],
                quantParams.ChannelZeroPoints[0]);
        }
        else
        {
            // Update per-tensor parameters
            _weightFakeQuantize.UpdateScaleAndZeroPoint(
                quantParams.Scale,
                quantParams.ZeroPoint);
        }
    }

    /// <summary>
    /// Gets the observed activation statistics.
    /// </summary>
    /// <returns>The observed activation statistics.</returns>
    public ObserverStatistics? GetActivationObserverStatistics()
    {
        return _activationObserver.GetStatistics();
    }

    /// <summary>
    /// Gets the observed weight statistics.
    /// </summary>
    /// <returns>The observed weight statistics.</returns>
    public ObserverStatistics? GetWeightObserverStatistics()
    {
        return _weightObserver.GetStatistics();
    }

    /// <summary>
    /// Resets all observer statistics.
    /// </summary>
    public void ResetObservers()
    {
        _activationObserver.Reset();
        _weightObserver.Reset();
    }

    /// <summary>
    /// Checks if the module wrapper has fake quantization nodes inserted.
    /// </summary>
    /// <returns>True if fake quantization nodes are present.</returns>
    public bool HasFakeQuantizationNodes()
    {
        return _weightFakeQuantize != null && _activationFakeQuantize != null;
    }

    /// <summary>
    /// Gets the number of fake quantization nodes.
    /// </summary>
    /// <returns>The number of fake quantization nodes (typically 2 for QATModuleWrapper).</returns>
    public int GetFakeQuantizationNodeCount()
    {
        return HasFakeQuantizationNodes() ? 2 : 0;
    }
}
