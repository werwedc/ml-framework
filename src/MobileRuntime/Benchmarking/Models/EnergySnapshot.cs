using System;

namespace MobileRuntime.Benchmarking.Models;

public class EnergySnapshot
{
    public double EnergyJoules { get; set; }
    public double PowerWatts { get; set; }
    public double VoltageVolts { get; set; }
    public double CurrentAmperes { get; set; }
    public DateTime Timestamp { get; set; }
}
