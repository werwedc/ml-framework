namespace MLFramework.Quantization.Evaluation;

/// <summary>
/// Detects accuracy regression between current and baseline reports.
/// </summary>
public class RegressionDetector
{
    private readonly float _regressionThreshold;

    /// <summary>
    /// Initializes a new instance of the RegressionDetector.
    /// </summary>
    /// <param name="regressionThreshold">Threshold for detecting regression</param>
    public RegressionDetector(float regressionThreshold = 0.01f)
    {
        _regressionThreshold = regressionThreshold;
    }

    /// <summary>
    /// Detects if there's an accuracy regression between current and baseline.
    /// </summary>
    /// <param name="current">Current accuracy report</param>
    /// <param name="baseline">Baseline accuracy report</param>
    /// <returns>True if regression detected</returns>
    public bool DetectRegression(AccuracyReport current, AccuracyReport baseline)
    {
        if (current == null)
            throw new ArgumentNullException(nameof(current));

        if (baseline == null)
            throw new ArgumentNullException(nameof(baseline));

        // Check if quantized accuracy dropped significantly
        float accuracyDelta = GetRegressionDelta(current, baseline);
        return accuracyDelta > _regressionThreshold;
    }

    /// <summary>
    /// Gets the regression amount between current and baseline.
    /// </summary>
    /// <param name="current">Current accuracy report</param>
    /// <param name="baseline">Baseline accuracy report</param>
    /// <returns>Regression delta (positive means regression)</returns>
    public float GetRegressionDelta(AccuracyReport current, AccuracyReport baseline)
    {
        if (current == null)
            throw new ArgumentNullException(nameof(current));

        if (baseline == null)
            throw new ArgumentNullException(nameof(baseline));

        // Calculate regression as the increase in accuracy drop
        float currentDrop = current.AccuracyDrop;
        float baselineDrop = baseline.AccuracyDrop;

        return currentDrop - baselineDrop;
    }

    /// <summary>
    /// Generates a detailed regression report.
    /// </summary>
    /// <param name="current">Current accuracy report</param>
    /// <param name="baseline">Baseline accuracy report</param>
    /// <returns>Formatted regression report</returns>
    public string GenerateRegressionReport(AccuracyReport current, AccuracyReport baseline)
    {
        if (current == null)
            throw new ArgumentNullException(nameof(current));

        if (baseline == null)
            throw new ArgumentNullException(nameof(baseline));

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Regression Detection Report");
        sb.AppendLine(new string('=', 60));
        sb.AppendLine();
        sb.AppendLine($"Baseline Timestamp: {baseline.Timestamp:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"Current Timestamp:  {current.Timestamp:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine();

        // FP32 comparison
        sb.AppendLine("FP32 Accuracy:");
        sb.AppendLine($"  Baseline: {baseline.FP32Accuracy:F4}");
        sb.AppendLine($"  Current:  {current.FP32Accuracy:F4}");
        sb.AppendLine($"  Delta:    {current.FP32Accuracy - baseline.FP32Accuracy:+0.0000;-0.0000}");
        sb.AppendLine();

        // Quantized comparison
        sb.AppendLine("Quantized Accuracy:");
        sb.AppendLine($"  Baseline: {baseline.QuantizedAccuracy:F4}");
        sb.AppendLine($"  Current:  {current.QuantizedAccuracy:F4}");
        sb.AppendLine($"  Delta:    {current.QuantizedAccuracy - baseline.QuantizedAccuracy:+0.0000;-0.0000}");
        sb.AppendLine();

        // Accuracy drop comparison
        float regressionDelta = GetRegressionDelta(current, baseline);
        sb.AppendLine("Accuracy Drop:");
        sb.AppendLine($"  Baseline: {baseline.AccuracyDrop:F4}");
        sb.AppendLine($"  Current:  {current.AccuracyDrop:F4}");
        sb.AppendLine($"  Delta:    {regressionDelta:+0.0000;-0.0000}");
        sb.AppendLine();

        // Regression status
        bool hasRegression = DetectRegression(current, baseline);
        sb.AppendLine($"Regression Detected: {hasRegression}");

        if (hasRegression)
        {
            sb.AppendLine($"  Regression amount: {regressionDelta:F4} exceeds threshold of {_regressionThreshold:F4}");
        }
        else
        {
            sb.AppendLine($"  No significant regression detected");
        }

        // Check per-layer changes
        if (baseline.PerLayerResults.Length > 0 && current.PerLayerResults.Length > 0)
        {
            sb.AppendLine();
            sb.AppendLine("Per-Layer Analysis:");

            foreach (var baselineLayer in baseline.PerLayerResults)
            {
                var currentLayer = current.PerLayerResults
                    .FirstOrDefault(l => l.LayerName == baselineLayer.LayerName);

                if (currentLayer.LayerName != null)
                {
                    float impactDelta = currentLayer.AccuracyImpact - baselineLayer.AccuracyImpact;
                    sb.AppendLine($"  {baselineLayer.LayerName}:");
                    sb.AppendLine($"    Baseline Impact: {baselineLayer.AccuracyImpact:F4}");
                    sb.AppendLine($"    Current Impact:  {currentLayer.AccuracyImpact:F4}");
                    sb.AppendLine($"    Delta:           {impactDelta:+0.0000;-0.0000}");

                    if (impactDelta > _regressionThreshold)
                    {
                        sb.AppendLine($"    ⚠ WARNING: Increased sensitivity detected!");
                    }
                }
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Gets the regression threshold.
    /// </summary>
    public float RegressionThreshold => _regressionThreshold;
}
