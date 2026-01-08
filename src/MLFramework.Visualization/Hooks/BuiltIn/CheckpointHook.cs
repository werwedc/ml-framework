using System;
using System.IO;

namespace MLFramework.Visualization.Hooks;

/// <summary>
/// Hook that periodically saves model checkpoints during training.
/// Checkpoints can be saved based on epochs or steps.
/// </summary>
public class CheckpointHook : ITrainingHook
{
    private int _epochsSinceLastSave = 0;
    private int _stepsSinceLastSave = 0;
    private float BestMetricValue;

    /// <summary>
    /// Creates a new checkpoint hook
    /// </summary>
    public CheckpointHook(string checkpointDirectory, int saveEveryNEpochs = 1, int saveEveryNSteps = 0)
    {
        CheckpointDirectory = checkpointDirectory ?? throw new ArgumentNullException(nameof(checkpointDirectory));
        SaveEveryNEpochs = saveEveryNEpochs;
        SaveEveryNSteps = saveEveryNSteps;
        BestMetricValue = MetricHigherIsBetter ? float.MinValue : float.MaxValue;

        if (!Directory.Exists(CheckpointDirectory))
        {
            Directory.CreateDirectory(CheckpointDirectory);
        }
    }

    public string CheckpointDirectory { get; set; }
    public int SaveEveryNEpochs { get; set; }
    public int SaveEveryNSteps { get; set; }
    public bool SaveBestOnly { get; set; }
    public string? BestMetricName { get; set; }
    public bool MetricHigherIsBetter { get; set; }
    public int MaxCheckpointsToKeep { get; set; } = 5;
    public Action<TrainingContext, string>? SaveCheckpointCallback { get; set; }

    public void OnPhaseStart(TrainingPhase phase, TrainingContext context) { }

    public void OnPhaseEnd(TrainingPhase phase, TrainingContext context)
    {
        switch (phase)
        {
            case TrainingPhase.EpochEnd:
                _epochsSinceLastSave++;
                if (SaveEveryNEpochs > 0 && _epochsSinceLastSave >= SaveEveryNEpochs)
                {
                    SaveCheckpoint(context);
                    _epochsSinceLastSave = 0;
                }
                break;

            case TrainingPhase.BatchEnd:
                _stepsSinceLastSave++;
                if (SaveEveryNSteps > 0 && _stepsSinceLastSave >= SaveEveryNSteps)
                {
                    SaveCheckpoint(context);
                    _stepsSinceLastSave = 0;
                }
                break;
        }
    }

    public void OnMetricUpdate(string metricName, float value, TrainingContext context) { }

    public void OnException(Exception exception, TrainingContext context) { }

    private void SaveCheckpoint(TrainingContext context)
    {
        if (SaveBestOnly && !string.IsNullOrEmpty(BestMetricName))
        {
            if (context.Metrics.TryGetValue(BestMetricName, out float metricValue))
            {
                bool isBetter = MetricHigherIsBetter
                    ? metricValue > BestMetricValue
                    : metricValue < BestMetricValue;

                if (!isBetter) return;

                BestMetricValue = metricValue;
            }
        }

        string filename = $"checkpoint_epoch{context.CurrentEpoch}_step{context.CurrentStep}.pt";
        string filepath = Path.Combine(CheckpointDirectory, filename);
        SaveCheckpointCallback?.Invoke(context, filepath);
        Console.WriteLine($"[CheckpointHook] Saved checkpoint: {filepath}");
    }
}
