namespace MachineLearning.Distributed.Models;

/// <summary>
/// Represents the current cluster topology
/// </summary>
public class ClusterTopology
{
    public int WorldSize { get; set; }
    public List<WorkerId> Workers { get; set; } = new();
    public DateTime LastUpdated { get; set; }
    public int Epoch { get; set; }

    public ClusterTopology()
    {
        LastUpdated = DateTime.UtcNow;
    }

    public void AddWorker(WorkerId worker)
    {
        Workers.Add(worker);
        WorldSize = Workers.Count;
        LastUpdated = DateTime.UtcNow;
        Epoch++;
    }

    public void RemoveWorker(WorkerId worker)
    {
        Workers.Remove(worker);
        WorldSize = Workers.Count;
        LastUpdated = DateTime.UtcNow;
        Epoch++;
    }
}
