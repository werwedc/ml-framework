using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.QAT;

/// <summary>
/// Interface for preparing and managing quantization-aware training (QAT) models.
/// </summary>
public interface IQATPreparer
{
    /// <summary>
    /// Prepares a model for quantization-aware training.
    /// This involves inserting fake quantization nodes, configuring quantization parameters,
    /// and wrapping layers for QAT compatibility.
    /// </summary>
    /// <param name="model">The model to prepare for QAT.</param>
    /// <param name="config">Global quantization configuration.</param>
    /// <param name="perLayerConfig">Optional per-layer configuration overrides.</param>
    /// <returns>A QAT-prepared model that implements IQATModel.</returns>
    IQATModel PrepareForQAT(
        object model,
        QuantizationConfig config,
        Dictionary<string, QuantizationConfig>? perLayerConfig = null);

    /// <summary>
    /// Converts a trained QAT model to a fully quantized Int8 model for inference.
    /// This extracts the trained quantization parameters and removes fake quantization nodes.
    /// </summary>
    /// <param name="qatModel">The trained QAT model to convert.</param>
    /// <returns>A quantized model ready for deployment.</returns>
    object ConvertToQuantized(IQATModel qatModel);

    /// <summary>
    /// Gets QAT statistics from a model during or after training.
    /// This includes quantization parameter evolution, activation ranges, and layer information.
    /// </summary>
    /// <param name="qatModel">The QAT model to get statistics from.</param>
    /// <returns>QAT statistics including scales, zero-points, and layer information.</returns>
    QATStatistics GetQATStatistics(IQATModel qatModel);
}
