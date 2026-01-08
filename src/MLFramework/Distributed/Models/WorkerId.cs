namespace MachineLearning.Distributed.Models;

/// <summary>
/// Unique identifier for a worker in the cluster
/// </summary>
public record WorkerId
{
    public string Id { get; init; } = string.Empty;
    public string Hostname { get; init; } = string.Empty;
    public int Port { get; init; }

    public WorkerId(string id, string hostname, int port)
    {
        Id = id;
        Hostname = hostname;
        Port = port;
    }

    public override string ToString() => $"{Id}@{Hostname}:{Port}";
}
