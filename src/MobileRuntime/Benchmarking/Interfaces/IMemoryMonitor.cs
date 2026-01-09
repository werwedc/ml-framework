using MobileRuntime.Benchmarking.Models;

namespace MobileRuntime.Benchmarking.Interfaces;

public interface IMemoryMonitor
{
    void StartMonitoring();
    void StopMonitoring();
    MemorySnapshot GetSnapshot();
    void Reset();
}
