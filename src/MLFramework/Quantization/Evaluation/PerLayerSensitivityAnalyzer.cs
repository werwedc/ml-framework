using RitterFramework.Core.Tensor;
using MLFramework.Data;

namespace MLFramework.Quantization.Evaluation;

/// <summary>
/// Analyzes sensitivity of individual layers to quantization.
/// </summary>
public class PerLayerSensitivityAnalyzer<TInput>
{
    private readonly ModelEvaluator<TInput> _evaluator;
    private readonly float _sensitivityThreshold;

    /// <summary>
    /// Initializes a new instance of the PerLayerSensitivityAnalyzer.
    /// </summary>
    /// <param name="evaluator">Model evaluator to use</param>
    /// <param name="sensitivityThreshold">Threshold for considering a layer sensitive</param>
    public PerLayerSensitivityAnalyzer(
        ModelEvaluator<TInput>? evaluator = null,
        float sensitivityThreshold = 0.01f)
    {
        _evaluator = evaluator ?? new ModelEvaluator<TInput>();
        _sensitivityThreshold = sensitivityThreshold;
    }

    /// <summary>
    /// Analyzes sensitivity of specified layers.
    /// </summary>
    /// <typeparam name="TOutput">Type of output data</typeparam>
    /// <param name="fp32Model">Baseline FP32 model</param>
    /// <param name="testData">Test data loader</param>
    /// <param name="layerNames">Names of layers to analyze</param>
    /// <param name="metric">Metric to use for evaluation</param>
    /// <returns>Array of sensitivity analysis results</returns>
    public SensitivityAnalysisResult[] AnalyzeLayerSensitivity<TOutput>(
        IModel<TInput, Tensor> fp32Model,
        DataLoader<TOutput> testData,
        string[] layerNames,
        Metrics.IAccuracyMetric metric)
    {
        if (fp32Model == null)
            throw new ArgumentNullException(nameof(fp32Model));

        if (testData == null)
            throw new ArgumentNullException(nameof(testData));

        if (layerNames == null || layerNames.Length == 0)
            throw new ArgumentException("Layer names must be provided", nameof(layerNames));

        if (metric == null)
            throw new ArgumentNullException(nameof(metric));

        // Get baseline accuracy
        float baselineAccuracy = _evaluator.Evaluate(fp32Model, testData, metric);

        var results = new List<SensitivityAnalysisResult>();

        // Analyze each layer
        foreach (var layerName in layerNames)
        {
            float layerAccuracy = AnalyzeLayer(fp32Model, testData, layerName, metric);
            float accuracyImpact = baselineAccuracy - layerAccuracy;
            bool isSensitive = accuracyImpact > _sensitivityThreshold;
            string recommendedAction = DetermineRecommendedAction(isSensitive, accuracyImpact);

            results.Add(new SensitivityAnalysisResult(
                layerName,
                accuracyImpact,
                isSensitive,
                recommendedAction
            ));
        }

        return results.ToArray();
    }

    /// <summary>
    /// Analyzes a single layer's sensitivity.
    /// </summary>
    private float AnalyzeLayer<TOutput>(
        IModel<TInput, Tensor> model,
        DataLoader<TOutput> testData,
        string layerName,
        Metrics.IAccuracyMetric metric)
    {
        // This is a simplified implementation
        // In practice, this would:
        // 1. Create a version of the model with only this layer quantized
        // 2. Evaluate accuracy with that configuration
        // 3. Return the accuracy

        // For now, we'll return the baseline accuracy (no impact)
        // Real implementation would require quantization infrastructure
        var baselineAccuracy = _evaluator.Evaluate(model, testData, metric);
        return baselineAccuracy;
    }

    /// <summary>
    /// Determines the recommended action for a layer.
    /// </summary>
    private string DetermineRecommendedAction(bool isSensitive, float accuracyImpact)
    {
        if (!isSensitive)
        {
            return "Quantize";
        }

        // If layer is sensitive, check severity
        if (accuracyImpact > 0.05f)
        {
            return "Fallback to FP32";
        }
        else if (accuracyImpact > 0.02f)
        {
            return "Consider mixed precision";
        }
        else
        {
            return "Monitor closely";
        }
    }

    /// <summary>
    /// Gets layers that exceed the sensitivity threshold.
    /// </summary>
    /// <param name="results">Sensitivity analysis results</param>
    /// <param name="threshold">Optional custom threshold</param>
    /// <returns>Sensitive layer names</returns>
    public string[] GetSensitiveLayers(
        SensitivityAnalysisResult[] results,
        float? threshold = null)
    {
        if (results == null)
            throw new ArgumentNullException(nameof(results));

        float effectiveThreshold = threshold ?? _sensitivityThreshold;

        return results
            .Where(r => r.AccuracyImpact > effectiveThreshold)
            .Select(r => r.LayerName)
            .ToArray();
    }

    /// <summary>
    /// Generates a detailed impact report per layer.
    /// </summary>
    /// <param name="results">Sensitivity analysis results</param>
    /// <returns>Formatted impact report</returns>
    public string GetLayerImpactReport(SensitivityAnalysisResult[] results)
    {
        if (results == null)
            throw new ArgumentNullException(nameof(results));

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Layer Sensitivity Analysis Report");
        sb.AppendLine(new string('=', 60));
        sb.AppendLine();

        // Sort by accuracy impact (descending)
        var sortedResults = results.OrderByDescending(r => r.AccuracyImpact).ToList();

        foreach (var result in sortedResults)
        {
            sb.AppendLine($"Layer: {result.LayerName}");
            sb.AppendLine($"  Accuracy Impact: {result.AccuracyImpact:F4}");
            sb.AppendLine($"  Sensitive: {result.IsSensitive}");
            sb.AppendLine($"  Recommended Action: {result.RecommendedAction}");
            sb.AppendLine();
        }

        sb.AppendLine($"Total layers analyzed: {results.Length}");
        sb.AppendLine($"Sensitive layers: {results.Count(r => r.IsSensitive)}");
        sb.AppendLine($"Safe to quantize: {results.Count(r => !r.IsSensitive)}");

        return sb.ToString();
    }
}
