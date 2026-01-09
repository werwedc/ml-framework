using System;
using System.Collections.Generic;

namespace MobileRuntime.Benchmarking.Models;

public class ProfileReport
{
    public List<ProfileEntry> Entries { get; set; } = new List<ProfileEntry>();
    public TimeSpan TotalTime { get; set; }
    public DateTime Timestamp { get; set; }
}
