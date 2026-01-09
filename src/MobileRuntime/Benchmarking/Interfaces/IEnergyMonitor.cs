using MobileRuntime.Benchmarking.Models;

namespace MobileRuntime.Benchmarking.Interfaces;

public interface IEnergyMonitor
{
    void StartMonitoring();
    void StopMonitoring();
    EnergySnapshot GetSnapshot();
    void Reset();
}
