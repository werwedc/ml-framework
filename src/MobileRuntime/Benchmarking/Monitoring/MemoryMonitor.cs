using System;
using System.Collections.Generic;
using System.Threading;
using MobileRuntime.Benchmarking.Interfaces;
using MobileRuntime.Benchmarking.Models;
using System.Diagnostics;

namespace MobileRuntime.Benchmarking.Monitoring;

public class MemoryMonitor : IMemoryMonitor
{
    private readonly Timer _monitorTimer;
    private readonly List<MemorySnapshot> _snapshots;
    private readonly object _lock = new object();
    private bool _isMonitoring;

    public MemoryMonitor(int sampleIntervalMs = 100)
    {
        _snapshots = new List<MemorySnapshot>();
        _monitorTimer = new Timer(SampleMemory, null, Timeout.Infinite, Timeout.Infinite);
    }

    public void StartMonitoring()
    {
        if (_isMonitoring)
            return;

        _isMonitoring = true;
        _monitorTimer.Change(0, 100);
    }

    public void StopMonitoring()
    {
        if (!_isMonitoring)
            return;

        _isMonitoring = false;
        _monitorTimer.Change(Timeout.Infinite, Timeout.Infinite);
    }

    public MemorySnapshot GetSnapshot()
    {
        lock (_lock)
        {
            if (_snapshots.Count == 0)
                return GetCurrentSnapshot();

            return _snapshots[_snapshots.Count - 1];
        }
    }

    public void Reset()
    {
        lock (_lock)
        {
            _snapshots.Clear();
        }
    }

    private void SampleMemory(object? state)
    {
        if (!_isMonitoring)
            return;

        var snapshot = GetCurrentSnapshot();
        lock (_lock)
        {
            _snapshots.Add(snapshot);
        }
    }

    private MemorySnapshot GetCurrentSnapshot()
    {
        var process = Process.GetCurrentProcess();
        process.Refresh();

        return new MemorySnapshot
        {
            WorkingSetBytes = process.WorkingSet64,
            PrivateMemoryBytes = process.PrivateMemorySize64,
            GCMemoryBytes = GC.GetTotalMemory(false),
            Gen0Collections = GC.CollectionCount(0),
            Gen1Collections = GC.CollectionCount(1),
            Gen2Collections = GC.CollectionCount(2),
            Timestamp = DateTime.UtcNow
        };
    }
}
