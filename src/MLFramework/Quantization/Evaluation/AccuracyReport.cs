namespace MLFramework.Quantization.Evaluation;

/// <summary>
/// Comprehensive accuracy report comparing FP32 and quantized models.
/// </summary>
public class AccuracyReport
{
    /// <summary>
    /// Baseline FP32 model accuracy.
    /// </summary>
    public float FP32Accuracy { get; set; }

    /// <summary>
    /// Quantized model accuracy.
    /// </summary>
    public float QuantizedAccuracy { get; set; }

    /// <summary>
    /// Accuracy drop from FP32 to quantized.
    /// </summary>
    public float AccuracyDrop => FP32Accuracy - QuantizedAccuracy;

    /// <summary>
    /// Whether the accuracy drop is acceptable based on threshold.
    /// </summary>
    public bool IsAcceptable { get; set; }

    /// <summary>
    /// Threshold used to determine if accuracy drop is acceptable.
    /// </summary>
    public float AcceptableThreshold { get; set; } = 0.01f;

    /// <summary>
    /// Per-layer sensitivity analysis results.
    /// </summary>
    public SensitivityAnalysisResult[] PerLayerResults { get; set; } = Array.Empty<SensitivityAnalysisResult>();

    /// <summary>
    /// Timestamp when the report was generated.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Additional metadata.
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();

    /// <summary>
    /// Gets sensitive layers (layers exceeding threshold).
    /// </summary>
    public SensitivityAnalysisResult[] GetSensitiveLayers()
    {
        return PerLayerResults.Where(r => r.IsSensitive).ToArray();
    }

    /// <summary>
    /// Gets recommended layers to keep in FP32.
    /// </summary>
    public string[] GetRecommendedFP32Layers()
    {
        return GetSensitiveLayers()
            .Where(r => r.RecommendedAction.Equals("Fallback to FP32", StringComparison.OrdinalIgnoreCase))
            .Select(r => r.LayerName)
            .ToArray();
    }

    /// <summary>
    /// Gets layers that can be safely quantized.
    /// </summary>
    public string[] GetQuantizableLayers()
    {
        return PerLayerResults
            .Where(r => !r.IsSensitive || r.RecommendedAction.Equals("Quantize", StringComparison.OrdinalIgnoreCase))
            .Select(r => r.LayerName)
            .ToArray();
    }
}
