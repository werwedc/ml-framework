using System;
using System.Collections.Generic;

namespace MLFramework.Visualization.Hooks;

/// <summary>
/// Hook that stops training early when a monitored metric stops improving.
/// Useful for preventing overfitting and saving training time.
/// </summary>
public class EarlyStoppingHook : ITrainingHook
{
    private readonly List<float> _metricHistory = new();
    private float BestMetricValue;

    public EarlyStoppingHook(string monitor = "loss", int patience = 5, float minDelta = 0.001f)
    {
        Monitor = monitor ?? throw new ArgumentNullException(nameof(monitor));
        Patience = patience;
        MinDelta = minDelta;
        BestMetricValue = MetricHigherIsBetter ? float.MinValue : float.MaxValue;
    }

    public string Monitor { get; set; }
    public int Patience { get; set; }
    public float MinDelta { get; set; }
    public bool MetricHigherIsBetter { get; set; }
    public bool RestoreBestWeights { get; set; }
    public Action? StopTrainingCallback { get; set; }
    public Action<TrainingContext>? RestoreWeightsCallback { get; set; }

    private int _epochsSinceImprovement = 0;
    private int BestEpoch = -1;
    public bool ShouldStop { get; private set; }

    public void OnPhaseStart(TrainingPhase phase, TrainingContext context) { }

    public void OnPhaseEnd(TrainingPhase phase, TrainingContext context)
    {
        if (phase != TrainingPhase.EpochEnd) return;

        if (!context.Metrics.TryGetValue(Monitor, out float currentMetricValue))
        {
            Console.WriteLine($"[EarlyStoppingHook] Metric '{Monitor}' not found in context");
            return;
        }

        _metricHistory.Add(currentMetricValue);

        bool isImprovement = MetricHigherIsBetter
            ? currentMetricValue > BestMetricValue + MinDelta
            : currentMetricValue < BestMetricValue - MinDelta;

        if (isImprovement)
        {
            BestMetricValue = currentMetricValue;
            BestEpoch = context.CurrentEpoch;
            _epochsSinceImprovement = 0;
            Console.WriteLine($"[EarlyStoppingHook] New best {Monitor}: {BestMetricValue:F4} at epoch {context.CurrentEpoch}");
        }
        else
        {
            _epochsSinceImprovement++;
            Console.WriteLine($"[EarlyStoppingHook] No improvement for {_epochsSinceImprovement} epochs");

            if (_epochsSinceImprovement >= Patience)
            {
                ShouldStop = true;
                Console.WriteLine($"[EarlyStoppingHook] Early stopping triggered at epoch {context.CurrentEpoch}");
                StopTrainingCallback?.Invoke();
            }
        }
    }

    public void OnMetricUpdate(string metricName, float value, TrainingContext context) { }

    public void OnException(Exception exception, TrainingContext context) { }

    public void Reset()
    {
        _metricHistory.Clear();
        _epochsSinceImprovement = 0;
        BestMetricValue = MetricHigherIsBetter ? float.MinValue : float.MaxValue;
        BestEpoch = -1;
        ShouldStop = false;
    }
}
