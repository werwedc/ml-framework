namespace MachineLearning.Distributed.Models;

using MachineLearning.Distributed.Enums;

/// <summary>
/// Event representing a topology change requiring rescaling
/// </summary>
public class RescalingEvent
{
    public RescaleType Type { get; set; }
    public ClusterTopology OldTopology { get; set; } = null!;
    public ClusterTopology NewTopology { get; set; } = null!;
    public List<WorkerId> AddedWorkers { get; set; } = new();
    public List<WorkerId> RemovedWorkers { get; set; } = new();
    public DateTime EventTime { get; set; }
    public string TriggerReason { get; set; } = string.Empty;

    public RescalingEvent()
    {
        EventTime = DateTime.UtcNow;
    }
}
