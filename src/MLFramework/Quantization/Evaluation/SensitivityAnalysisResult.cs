namespace MLFramework.Quantization.Evaluation;

/// <summary>
/// Result of per-layer sensitivity analysis.
/// </summary>
public struct SensitivityAnalysisResult
{
    /// <summary>
    /// Name of the layer.
    /// </summary>
    public string LayerName { get; set; }

    /// <summary>
    /// Accuracy impact when this layer is quantized.
    /// </summary>
    public float AccuracyImpact { get; set; }

    /// <summary>
    /// Whether this layer is sensitive to quantization.
    /// </summary>
    public bool IsSensitive { get; set; }

    /// <summary>
    /// Recommended action for this layer.
    /// </summary>
    public string RecommendedAction { get; set; }

    /// <summary>
    /// Creates a new SensitivityAnalysisResult.
    /// </summary>
    public SensitivityAnalysisResult(
        string layerName,
        float accuracyImpact,
        bool isSensitive,
        string recommendedAction)
    {
        LayerName = layerName;
        AccuracyImpact = accuracyImpact;
        IsSensitive = isSensitive;
        RecommendedAction = recommendedAction;
    }
}
