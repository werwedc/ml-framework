using System;

namespace MobileRuntime.Benchmarking.Models;

public class MemorySnapshot
{
    public long WorkingSetBytes { get; set; }
    public long PrivateMemoryBytes { get; set; }
    public long GCMemoryBytes { get; set; }
    public long Gen0Collections { get; set; }
    public long Gen1Collections { get; set; }
    public long Gen2Collections { get; set; }
    public DateTime Timestamp { get; set; }
}
