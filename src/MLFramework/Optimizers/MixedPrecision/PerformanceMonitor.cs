using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace MLFramework.Optimizers.MixedPrecision;

/// <summary>
/// Monitors and tracks performance metrics for mixed-precision training
/// </summary>
public class PerformanceMonitor
{
    private readonly MixedPrecisionOptions _options;
    private readonly Stopwatch _stepTimer;
    private readonly Queue<float> _stepTimes;
    private readonly int _maxHistorySize;

    private long _stepStartTimeTicks;
    private long _totalElapsedTicks;
    private int _stepCount;

    #region Properties

    /// <summary>
    /// Average step time in milliseconds
    /// </summary>
    public float AverageStepTimeMs { get; private set; }

    /// <summary>
    /// Last step time in milliseconds
    /// </summary>
    public float LastStepTimeMs { get; private set; }

    /// <summary>
    /// Minimum step time in milliseconds
    /// </summary>
    public float MinStepTimeMs { get; private set; } = float.MaxValue;

    /// <summary>
    /// Maximum step time in milliseconds
    /// </summary>
    public float MaxStepTimeMs { get; private set; } = float.MinValue;

    /// <summary>
    /// Total training time in seconds
    /// </summary>
    public float TotalTimeSeconds => _totalElapsedTicks / 10000000.0f;

    /// <summary>
    /// Number of steps tracked
    /// </summary>
    public int StepCount => _stepCount;

    /// <summary>
    /// Estimated training speed (steps per second)
    /// </summary>
    public float StepsPerSecond => _stepCount > 0 ? 1000.0f / AverageStepTimeMs : 0;

    #endregion

    #region Constructors

    public PerformanceMonitor(MixedPrecisionOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        options.Validate();

        _stepTimer = new Stopwatch();
        _stepTimes = new Queue<float>();
        _maxHistorySize = options.PerformanceLogInterval;

        AverageStepTimeMs = 0;
        LastStepTimeMs = 0;
        _totalElapsedTicks = 0;
        _stepCount = 0;
    }

    public PerformanceMonitor()
        : this(MixedPrecisionOptions.ForFP16())
    {
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Starts timing a training step
    /// </summary>
    public void StartStep()
    {
        _stepTimer.Restart();
        _stepStartTimeTicks = Stopwatch.GetTimestamp();
    }

    /// <summary>
    /// Ends timing a training step
    /// </summary>
    public void EndStep()
    {
        _stepTimer.Stop();
        float elapsedMs = _stepTimer.ElapsedMilliseconds;

        _totalElapsedTicks += _stepTimer.ElapsedTicks;
        _stepCount++;
        LastStepTimeMs = elapsedMs;

        // Update statistics
        UpdateStatistics(elapsedMs);

        // Log periodically
        if (_stepCount % _options.PerformanceLogInterval == 0)
        {
            LogPerformance();
        }
    }

    /// <summary>
    /// Gets current performance statistics
    /// </summary>
    public PerformanceStats GetStats()
    {
        return new PerformanceStats
        {
            AverageStepTimeMs = AverageStepTimeMs,
            LastStepTimeMs = LastStepTimeMs,
            MinStepTimeMs = MinStepTimeMs == float.MaxValue ? 0 : MinStepTimeMs,
            MaxStepTimeMs = MaxStepTimeMs == float.MinValue ? 0 : MaxStepTimeMs,
            TotalTimeSeconds = TotalTimeSeconds,
            StepCount = _stepCount,
            StepsPerSecond = StepsPerSecond
        };
    }

    /// <summary>
    /// Resets all statistics
    /// </summary>
    public void Reset()
    {
        _stepTimer.Reset();
        _stepTimes.Clear();
        _totalElapsedTicks = 0;
        _stepCount = 0;
        AverageStepTimeMs = 0;
        LastStepTimeMs = 0;
        MinStepTimeMs = float.MaxValue;
        MaxStepTimeMs = float.MinValue;
    }

    /// <summary>
    /// Gets a summary string of current performance
    /// </summary>
    public string GetSummary()
    {
        var stats = GetStats();
        return $"Steps: {stats.StepCount}, " +
               $"Avg: {stats.AverageStepTimeMs:F2}ms, " +
               $"Last: {stats.LastStepTimeMs:F2}ms, " +
               $"Speed: {stats.StepsPerSecond:F2} steps/sec";
    }

    #endregion

    #region Private Methods

    private void UpdateStatistics(float elapsedMs)
    {
        // Update min/max
        MinStepTimeMs = Math.Min(MinStepTimeMs, elapsedMs);
        MaxStepTimeMs = Math.Max(MaxStepTimeMs, elapsedMs);

        // Add to history
        _stepTimes.Enqueue(elapsedMs);
        if (_stepTimes.Count > _maxHistorySize)
        {
            _stepTimes.Dequeue();
        }

        // Update running average
        AverageStepTimeMs = ComputeRunningAverage(_stepTimes);
    }

    private float ComputeRunningAverage(Queue<float> values)
    {
        if (values.Count == 0)
            return 0;

        float sum = 0;
        foreach (var value in values)
        {
            sum += value;
        }

        return sum / values.Count;
    }

    private void LogPerformance()
    {
        Console.WriteLine($"[PerformanceMonitor] {GetSummary()}");
    }

    #endregion
}

/// <summary>
/// Performance statistics snapshot
/// </summary>
public class PerformanceStats
{
    public float AverageStepTimeMs { get; set; }
    public float LastStepTimeMs { get; set; }
    public float MinStepTimeMs { get; set; }
    public float MaxStepTimeMs { get; set; }
    public float TotalTimeSeconds { get; set; }
    public int StepCount { get; set; }
    public float StepsPerSecond { get; set; }

    /// <summary>
    /// Standard deviation of step times (if available)
    /// </summary>
    public float StdDevStepTimeMs { get; set; }

    public override string ToString()
    {
        return $"Avg: {AverageStepTimeMs:F2}ms, " +
               $"Range: [{MinStepTimeMs:F2}, {MaxStepTimeMs:F2}]ms, " +
               $"Speed: {StepsPerSecond:F2} steps/sec, " +
               $"Total: {TotalTimeSeconds:F2}s";
    }
}

/// <summary>
/// Memory usage statistics (placeholder for future enhancement)
/// </summary>
public class MemoryStats
{
    /// <summary>
    /// Peak GPU memory usage in MB
    /// </summary>
    public float PeakGPUMemoryMB { get; set; }

    /// <summary>
    /// Current GPU memory usage in MB
    /// </summary>
    public float CurrentGPUMemoryMB { get; set; }

    /// <summary>
    /// Memory savings from mixed precision in MB
    /// </summary>
    public float MemorySavingsMB { get; set; }

    /// <summary>
    /// Memory savings as percentage
    /// </summary>
    public float MemorySavingsPercent { get; set; }

    public override string ToString()
    {
        return $"GPU: {CurrentGPUMemoryMB:F2}MB, " +
               $"Peak: {PeakGPUMemoryMB:F2}MB, " +
               $"Saved: {MemorySavingsMB:F2}MB ({MemorySavingsPercent:P1})";
    }
}
