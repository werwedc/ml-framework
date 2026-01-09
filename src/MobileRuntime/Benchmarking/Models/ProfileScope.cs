using System;
using System.Diagnostics;

namespace MobileRuntime.Benchmarking.Models;

public class ProfileScope : IDisposable
{
    private readonly Action<string, TimeSpan> _onComplete;
    private readonly string _name;
    private readonly Stopwatch _stopwatch;

    internal ProfileScope(string name, Action<string, TimeSpan> onComplete)
    {
        _name = name;
        _onComplete = onComplete;
        _stopwatch = Stopwatch.StartNew();
    }

    public void Dispose()
    {
        _stopwatch.Stop();
        _onComplete?.Invoke(_name, _stopwatch.Elapsed);
    }
}
