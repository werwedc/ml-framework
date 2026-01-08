namespace MachineLearning.Distributed.Models;

/// <summary>
/// Global training state that needs to be synchronized across workers
/// </summary>
public class GlobalTrainingState
{
    public int CurrentEpoch { get; set; }
    public int CurrentStep { get; set; }
    public float LearningRate { get; set; }
    public int GlobalBatchSize { get; set; }
    public int ActiveWorkerCount { get; set; }
    public DateTime StateTimestamp { get; set; }

    // Optional: serialized optimizer state
    public byte[]? OptimizerState { get; set; }

    public GlobalTrainingState()
    {
        StateTimestamp = DateTime.UtcNow;
    }
}
