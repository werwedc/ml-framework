namespace MachineLearning.Distributed.Models;

using MachineLearning.Distributed.Enums;

/// <summary>
/// Metadata information about a worker
/// </summary>
public class WorkerMetadata
{
    public WorkerId WorkerId { get; set; } = null!;
    public WorkerStatus Status { get; set; }
    public DateTime JoinTime { get; set; }
    public DateTime LastHeartbeat { get; set; }
    public int Rank { get; set; }
    public int LocalWorldSize { get; set; }
    public Dictionary<string, string> CustomAttributes { get; set; } = new();

    public bool IsHealthy(TimeSpan timeout)
    {
        return DateTime.UtcNow - LastHeartbeat < timeout;
    }
}
