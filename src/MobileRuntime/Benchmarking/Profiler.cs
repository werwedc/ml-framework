using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MobileRuntime.Benchmarking.Models;
using MobileRuntime.Benchmarking.Export;

namespace MobileRuntime.Benchmarking;

public class Profiler : IDisposable
{
    private readonly Dictionary<string, ProfileEntry> _profiles;
    private readonly Stack<ProfileScope> _activeScopes;
    private readonly Stopwatch _stopwatch;
    private bool _disposed;

    public Profiler()
    {
        _profiles = new Dictionary<string, ProfileEntry>();
        _activeScopes = new Stack<ProfileScope>();
        _stopwatch = new Stopwatch();
        _stopwatch.Start();
    }

    public ProfileScope BeginScope(string name)
    {
        var scope = new ProfileScope(name, EndScope);
        _activeScopes.Push(scope);
        return scope;
    }

    public void EndScope()
    {
        if (_activeScopes.Count > 0)
        {
            var scope = _activeScopes.Pop();
            scope.Dispose();
        }
    }

    internal void EndScope(string name, TimeSpan elapsed)
    {
        if (!_profiles.ContainsKey(name))
        {
            _profiles[name] = new ProfileEntry
            {
                Name = name,
                CallCount = 0,
                TotalTime = TimeSpan.Zero,
                MinTime = elapsed,
                MaxTime = elapsed,
                AverageTime = elapsed,
                TotalMemoryBytes = 0
            };
        }

        var entry = _profiles[name];
        entry.CallCount++;
        entry.TotalTime += elapsed;

        if (elapsed < entry.MinTime)
            entry.MinTime = elapsed;

        if (elapsed > entry.MaxTime)
            entry.MaxTime = elapsed;

        entry.AverageTime = TimeSpan.FromMilliseconds(entry.TotalTime.TotalMilliseconds / entry.CallCount);
        entry.TotalMemoryBytes += GC.GetTotalMemory(false);

        _profiles[name] = entry;
    }

    public T Profile<T>(string name, Func<T> action)
    {
        using (BeginScope(name))
        {
            return action();
        }
    }

    public void Profile(string name, Action action)
    {
        using (BeginScope(name))
        {
            action();
        }
    }

    public ProfileReport GetReport()
    {
        return new ProfileReport
        {
            Entries = _profiles.Values.OrderByDescending(e => e.TotalTime).ToList(),
            TotalTime = _stopwatch.Elapsed,
            Timestamp = DateTime.UtcNow
        };
    }

    public void Reset()
    {
        _profiles.Clear();
        _activeScopes.Clear();
        _stopwatch.Restart();
    }

    public void ExportReport(string filePath, string format = ReportFormat.Json)
    {
        var report = GetReport();
        BenchmarkExporter.ExportProfile(report, filePath, format);
    }

    public void PrintReport()
    {
        var report = GetReport();
        Console.WriteLine($"Profiling Report - {report.Timestamp:yyyy-MM-dd HH:mm:ss}");
        Console.WriteLine($"Total Time: {report.TotalTime.TotalMilliseconds:F2}ms");
        Console.WriteLine();
        Console.WriteLine("Name".PadRight(30) + "Calls".PadRight(10) + "Total (ms)".PadRight(15) + "Avg (ms)".PadRight(15) + "Min (ms)".PadRight(15) + "Max (ms)".PadRight(15));
        Console.WriteLine(new string('-', 100));

        foreach (var entry in report.Entries)
        {
            Console.WriteLine(
                entry.Name.PadRight(30) +
                entry.CallCount.ToString().PadRight(10) +
                entry.TotalTime.TotalMilliseconds.ToString("F2").PadRight(15) +
                entry.AverageTime.TotalMilliseconds.ToString("F2").PadRight(15) +
                entry.MinTime.TotalMilliseconds.ToString("F2").PadRight(15) +
                entry.MaxTime.TotalMilliseconds.ToString("F2").PadRight(15)
            );
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            while (_activeScopes.Count > 0)
            {
                _activeScopes.Pop().Dispose();
            }

            _stopwatch.Stop();
            _disposed = true;
        }

        GC.SuppressFinalize(this);
    }
}
