using System;

namespace MLFramework.Visualization.Hooks;

/// <summary>
/// Hook that adjusts learning rate based on training metrics or schedule.
/// </summary>
public class LearningRateSchedulerHook : ITrainingHook
{
    public LearningRateSchedulerHook(Func<float> getLearningRate, Action<float> setLearningRate)
    {
        GetLearningRate = getLearningRate ?? throw new ArgumentNullException(nameof(getLearningRate));
        SetLearningRate = setLearningRate ?? throw new ArgumentNullException(nameof(setLearningRate));
    }

    public Func<float> GetLearningRate { get; set; }
    public Action<float> SetLearningRate { get; set; }
    public bool StepOnBatch { get; set; }
    public bool StepOnEpoch { get; set; } = true;
    public string? MonitorMetric { get; set; }
    public string Mode { get; set; } = "min";
    public float Factor { get; set; } = 0.1f;
    public float Threshold { get; set; } = 0.001f;
    public int Patience { get; set; } = 5;
    public float MinLearningRate { get; set; } = 1e-6f;

    private int _stepsSinceAdjustment = 0;
    private float BestMetricValue;
    private bool _initialized = false;
    public int ReductionCount { get; private set; }

    public void OnPhaseStart(TrainingPhase phase, TrainingContext context) { }

    public void OnPhaseEnd(TrainingPhase phase, TrainingContext context)
    {
        if ((StepOnBatch && phase == TrainingPhase.BatchEnd) ||
            (StepOnEpoch && phase == TrainingPhase.EpochEnd))
        {
            _stepsSinceAdjustment++;
            CheckAdjustment(context);
        }
    }

    public void OnMetricUpdate(string metricName, float value, TrainingContext context)
    {
        if (string.IsNullOrEmpty(MonitorMetric) || metricName != MonitorMetric) return;

        if (!_initialized)
        {
            BestMetricValue = value;
            _initialized = true;
            return;
        }

        bool improved;
        if (Mode == "min")
        {
            improved = value < BestMetricValue - Threshold;
            BestMetricValue = Math.Min(BestMetricValue, value);
        }
        else
        {
            improved = value > BestMetricValue + Threshold;
            BestMetricValue = Math.Max(BestMetricValue, value);
        }

        if (improved)
        {
            _stepsSinceAdjustment = 0;
        }
        else
        {
            _stepsSinceAdjustment++;
            CheckAdjustment(context);
        }
    }

    public void OnException(Exception exception, TrainingContext context) { }

    private void CheckAdjustment(TrainingContext context)
    {
        if (_stepsSinceAdjustment < Patience) return;

        float currentLR = GetLearningRate();
        float newLR = currentLR * Factor;

        if (newLR < MinLearningRate)
        {
            Console.WriteLine($"[LRSchedulerHook] Learning rate already at minimum");
            return;
        }

        SetLearningRate(newLR);
        ReductionCount++;
        Console.WriteLine($"[LRSchedulerHook] Reduced learning rate from {currentLR:F6} to {newLR:F6}");
        _stepsSinceAdjustment = 0;
    }

    public void Reset()
    {
        _stepsSinceAdjustment = 0;
        _initialized = false;
        BestMetricValue = Mode == "min" ? float.MaxValue : float.MinValue;
        ReductionCount = 0;
    }
}
