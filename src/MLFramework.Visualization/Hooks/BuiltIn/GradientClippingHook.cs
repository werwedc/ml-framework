using System;
using System.Collections.Generic;

namespace MLFramework.Visualization.Hooks;

/// <summary>
/// Hook that clips gradients to prevent exploding gradients during training.
/// </summary>
public class GradientClippingHook : ITrainingHook
{
    public GradientClippingHook(Action<float> clipGradientsCallback)
    {
        ClipGradientsCallback = clipGradientsCallback ?? throw new ArgumentNullException(nameof(clipGradientsCallback));
    }

    public Action<float> ClipGradientsCallback { get; set; }
    public string ClippingType { get; set; } = "norm";
    public float MaxClipValue { get; set; } = 1.0f;
    public int ClippingCount { get; private set; }
    public float TotalClippedNorm { get; private set; }
    public float MaxGradientNorm { get; private set; }
    public bool LogGradientStats { get; set; }

    public void OnPhaseStart(TrainingPhase phase, TrainingContext context) { }

    public void OnPhaseEnd(TrainingPhase phase, TrainingContext context)
    {
        if (phase != TrainingPhase.BackwardPassEnd) return;

        try
        {
            ClipGradientsCallback(MaxClipValue);
            ClippingCount++;

            if (LogGradientStats)
            {
                Console.WriteLine($"[GradientClippingHook] Clipped gradients ({ClippingType}, max={MaxClipValue:F2})");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[GradientClippingHook] Error: {ex.Message}");
        }
    }

    public void OnMetricUpdate(string metricName, float value, TrainingContext context) { }

    public void OnException(Exception exception, TrainingContext context) { }

    public void RecordGradientStatistics(float gradientNorm)
    {
        MaxGradientNorm = Math.Max(MaxGradientNorm, gradientNorm);

        if (gradientNorm > MaxClipValue)
        {
            TotalClippedNorm += gradientNorm - MaxClipValue;
        }
    }

    public Dictionary<string, float> GetStatistics()
    {
        return new Dictionary<string, float>
        {
            { "clipping_count", ClippingCount },
            { "total_clipped_norm", TotalClippedNorm },
            { "max_gradient_norm", MaxGradientNorm },
            { "avg_clipped_norm", ClippingCount > 0 ? TotalClippedNorm / ClippingCount : 0f }
        };
    }

    public void Reset()
    {
        ClippingCount = 0;
        TotalClippedNorm = 0f;
        MaxGradientNorm = 0f;
    }
}
