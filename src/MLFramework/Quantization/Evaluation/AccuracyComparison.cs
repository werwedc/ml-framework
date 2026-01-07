using RitterFramework.Core.Tensor;
using MLFramework.Data;

namespace MLFramework.Quantization.Evaluation;

/// <summary>
/// Compares accuracy between two models (e.g., FP32 vs quantized).
/// </summary>
public class AccuracyComparison<TInput>
{
    private readonly ModelEvaluator<TInput> _evaluator;
    private readonly float _acceptableThreshold;

    /// <summary>
    /// Initializes a new instance of the AccuracyComparison.
    /// </summary>
    /// <param name="evaluator">Model evaluator to use</param>
    /// <param name="acceptableThreshold">Threshold for acceptable accuracy drop</param>
    public AccuracyComparison(
        ModelEvaluator<TInput>? evaluator = null,
        float acceptableThreshold = 0.01f)
    {
        _evaluator = evaluator ?? new ModelEvaluator<TInput>();
        _acceptableThreshold = acceptableThreshold;
    }

    /// <summary>
    /// Compares two models on test data.
    /// </summary>
    /// <typeparam name="TOutput">Type of output data</typeparam>
    /// <param name="fp32Model">Baseline FP32 model</param>
    /// <param name="quantizedModel">Quantized model</param>
    /// <param name="testData">Test data loader</param>
    /// <param name="metrics">Metrics to compute</param>
    /// <returns>Accuracy comparison report</returns>
    public AccuracyReport CompareModels<TOutput>(
        IModel<TInput, Tensor> fp32Model,
        IModel<TInput, Tensor> quantizedModel,
        DataLoader<TOutput> testData,
        params Metrics.IAccuracyMetric[] metrics)
    {
        if (fp32Model == null)
            throw new ArgumentNullException(nameof(fp32Model));

        if (quantizedModel == null)
            throw new ArgumentNullException(nameof(quantizedModel));

        if (testData == null)
            throw new ArgumentNullException(nameof(testData));

        if (metrics == null || metrics.Length == 0)
            throw new ArgumentException("At least one metric must be provided", nameof(metrics));

        // Use the first metric as the primary accuracy metric
        var primaryMetric = metrics[0];

        // Evaluate both models
        float fp32Accuracy = _evaluator.Evaluate(fp32Model, testData, primaryMetric);
        float quantizedAccuracy = _evaluator.Evaluate(quantizedModel, testData, primaryMetric);

        // Create report
        var report = new AccuracyReport
        {
            FP32Accuracy = fp32Accuracy,
            QuantizedAccuracy = quantizedAccuracy,
            AcceptableThreshold = _acceptableThreshold,
            IsAcceptable = (fp32Accuracy - quantizedAccuracy) <= _acceptableThreshold,
            PerLayerResults = Array.Empty<SensitivityAnalysisResult>(),
            Metadata = new Dictionary<string, object>
            {
                ["FP32Model"] = fp32Model.Name,
                ["QuantizedModel"] = quantizedModel.Name,
                ["PrimaryMetric"] = primaryMetric.Name,
                ["HigherIsBetter"] = primaryMetric.HigherIsBetter
            }
        };

        return report;
    }

    /// <summary>
    /// Gets the accuracy delta between two models.
    /// </summary>
    /// <param name="report">Accuracy report</param>
    /// <returns>Accuracy difference</returns>
    public float GetAccuracyDelta(AccuracyReport report)
    {
        if (report == null)
            throw new ArgumentNullException(nameof(report));

        return report.AccuracyDrop;
    }

    /// <summary>
    /// Gets metrics per layer (if available).
    /// </summary>
    /// <param name="report">Accuracy report</param>
    /// <returns>Per-layer results</returns>
    public SensitivityAnalysisResult[] GetPerLayerMetrics(AccuracyReport report)
    {
        if (report == null)
            throw new ArgumentNullException(nameof(report));

        return report.PerLayerResults;
    }
}
