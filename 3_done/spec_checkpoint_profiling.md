# Spec: Profiling and Monitoring

## Overview
Implement profiling and monitoring tools for checkpointing, including real-time memory tracking, recomputation time analysis, and optimization recommendations.

## Classes

### Location
`src/MLFramework/Checkpointing/Profiling/`

### Class: CheckpointProfiler

```csharp
namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Profiles checkpointing operations and provides performance metrics
/// </summary>
public class CheckpointProfiler : IDisposable
{
    private readonly Dictionary<string, LayerProfile> _layerProfiles;
    private readonly List<CheckpointEvent> _events;
    private readonly CheckpointManager _checkpointManager;
    private readonly RecomputationEngine _recomputeEngine;
    private readonly object _lock = new object();
    private bool _isProfiling;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of CheckpointProfiler
    /// </summary>
    /// <param name="checkpointManager">Checkpoint manager to profile</param>
    /// <param name="recomputeEngine">Recompute engine to profile</param>
    public CheckpointProfiler(
        CheckpointManager checkpointManager,
        RecomputationEngine recomputeEngine)
    {
        _checkpointManager = checkpointManager ?? throw new ArgumentNullException(nameof(checkpointManager));
        _recomputeEngine = recomputeEngine ?? throw new ArgumentNullException(nameof(recomputeEngine));
        _layerProfiles = new Dictionary<string, LayerProfile>();
        _events = new List<CheckpointEvent>();
        _isProfiling = false;
        _disposed = false;
    }

    /// <summary>
    /// Starts profiling
    /// </summary>
    public void StartProfiling()
    {
        lock (_lock)
        {
            _isProfiling = true;
            _events.Clear();
            _layerProfiles.Clear();
        }
    }

    /// <summary>
    /// Stops profiling
    /// </summary>
    public void StopProfiling()
    {
        lock (_lock)
        {
            _isProfiling = false;
        }
    }

    /// <summary>
    /// Gets whether profiling is currently active
    /// </summary>
    public bool IsProfiling => _isProfiling;

    /// <summary>
    /// Records a checkpoint event
    /// </summary>
    /// <param name="layerId">Layer ID</param>
    /// <param name="eventType">Type of event</param>
    /// <param name="durationMs">Duration in milliseconds</param>
    /// <param name="memoryBytes">Memory affected in bytes</param>
    public void RecordEvent(
        string layerId,
        CheckpointEventType eventType,
        long durationMs,
        long memoryBytes = 0)
    {
        if (!_isProfiling)
            return;

        lock (_lock)
        {
            var @event = new CheckpointEvent
            {
                LayerId = layerId,
                EventType = eventType,
                Timestamp = DateTime.UtcNow,
                DurationMs = durationMs,
                MemoryBytes = memoryBytes
            };

            _events.Add(@event);

            // Update layer profile
            if (!_layerProfiles.ContainsKey(layerId))
            {
                _layerProfiles[layerId] = new LayerProfile { LayerId = layerId };
            }

            var profile = _layerProfiles[layerId];
            profile.RecordEvent(@event);
        }
    }

    /// <summary>
    /// Gets profile for a specific layer
    /// </summary>
    /// <param name="layerId">Layer ID</param>
    /// <returns>Layer profile or null if not found</returns>
    public LayerProfile? GetLayerProfile(string layerId)
    {
        lock (_lock)
        {
            return _layerProfiles.TryGetValue(layerId, out var profile) ? profile : null;
        }
    }

    /// <summary>
    /// Gets all layer profiles
    /// </summary>
    /// <returns>Dictionary of layer IDs to profiles</returns>
    public Dictionary<string, LayerProfile> GetAllLayerProfiles()
    {
        lock (_lock)
        {
            return new Dictionary<string, LayerProfile>(_layerProfiles);
        }
    }

    /// <summary>
    /// Gets profiling summary
    /// </summary>
    /// <returns>Profiling summary</returns>
    public ProfilingSummary GetSummary()
    {
        lock (_lock)
        {
            var summary = new ProfilingSummary
            {
                StartTime = _events.FirstOrDefault()?.Timestamp ?? DateTime.UtcNow,
                EndTime = _events.LastOrDefault()?.Timestamp ?? DateTime.UtcNow,
                TotalEvents = _events.Count,
                TotalCheckpointTime = _events
                    .Where(e => e.EventType == CheckpointEventType.Checkpoint)
                    .Sum(e => e.DurationMs),
                TotalRecomputeTime = _events
                    .Where(e => e.EventType == CheckpointEventType.Recompute)
                    .Sum(e => e.DurationMs),
                TotalMemorySaved = _events
                    .Where(e => e.EventType == CheckpointEventType.Checkpoint)
                    .Sum(e => e.MemoryBytes),
                LayerProfiles = _layerProfiles.Values.ToList()
            };

            summary.Duration = (summary.EndTime - summary.StartTime).TotalMilliseconds;

            return summary;
        }
    }

    /// <summary>
    /// Generates a profiling report
    /// </summary>
    /// <returns>Profiling report as string</returns>
    public string GenerateReport()
    {
        var summary = GetSummary();
        return summary.ToString();
    }

    /// <summary>
    /// Exports profiling data to JSON
    /// </summary>
    /// <returns>JSON string</returns>
    public string ExportToJson()
    {
        var summary = GetSummary();
        return JsonSerializer.Serialize(summary, new JsonSerializerOptions
        {
            WriteIndented = true
        });
    }

    /// <summary>
    /// Disposes the profiler
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            StopProfiling();
            _events.Clear();
            _layerProfiles.Clear();
            _disposed = true;
        }
    }
}
```

## Data Structures

### Enum: CheckpointEventType

```csharp
namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Types of checkpoint events
/// </summary>
public enum CheckpointEventType
{
    /// <summary>
    /// Checkpoint registration
    /// </summary>
    Checkpoint,

    /// <summary>
    /// Activation recomputation
    /// </summary>
    Recompute,

    /// <summary>
    /// Checkpoint retrieval (cache hit)
    /// </summary>
    Retrieve,

    /// <summary>
    /// Checkpoint deallocation
    /// </summary>
    Deallocate
}
```

### Class: CheckpointEvent

```csharp
namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Represents a checkpoint event
/// </summary>
public class CheckpointEvent
{
    /// <summary>
    /// Layer ID
    /// </summary>
    public string LayerId { get; set; } = string.Empty;

    /// <summary>
    /// Type of event
    /// </summary>
    public CheckpointEventType EventType { get; set; }

    /// <summary>
    /// Timestamp of the event
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Duration in milliseconds
    /// </summary>
    public long DurationMs { get; set; }

    /// <summary>
    /// Memory affected in bytes
    /// </summary>
    public long MemoryBytes { get; set; }
}
```

### Class: LayerProfile

```csharp
namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Profile for a specific layer
/// </summary>
public class LayerProfile
{
    /// <summary>
    /// Layer ID
    /// </summary>
    public string LayerId { get; set; } = string.Empty;

    /// <summary>
    /// Number of checkpoints
    /// </summary>
    public int CheckpointCount { get; set; }

    /// <summary>
    /// Total checkpoint time in milliseconds
    /// </summary>
    public long TotalCheckpointTimeMs { get; set; }

    /// <summary>
    /// Average checkpoint time in milliseconds
    /// </summary>
    public double AverageCheckpointTimeMs =>
        CheckpointCount > 0 ? (double)TotalCheckpointTimeMs / CheckpointCount : 0.0;

    /// <summary>
    /// Number of recomputations
    /// </summary>
    public int RecomputeCount { get; set; }

    /// <summary>
    /// Total recomputation time in milliseconds
    /// </summary>
    public long TotalRecomputeTimeMs { get; set; }

    /// <summary>
    /// Average recomputation time in milliseconds
    /// </summary>
    public double AverageRecomputeTimeMs =>
        RecomputeCount > 0 ? (double)TotalRecomputeTimeMs / RecomputeCount : 0.0;

    /// <summary>
    /// Number of cache hits
    /// </summary>
    public int CacheHitCount { get; set; }

    /// <summary>
    /// Cache hit rate
    /// </summary>
    public double CacheHitRate =>
        CheckpointCount > 0 ? (double)CacheHitCount / CheckpointCount : 0.0;

    /// <summary>
    /// Total memory saved in bytes
    /// </summary>
    public long TotalMemorySaved { get; set; }

    /// <summary>
    /// Records an event
    /// </summary>
    public void RecordEvent(CheckpointEvent @event)
    {
        switch (@event.EventType)
        {
            case CheckpointEventType.Checkpoint:
                CheckpointCount++;
                TotalCheckpointTimeMs += @event.DurationMs;
                TotalMemorySaved += @event.MemoryBytes;
                break;
            case CheckpointEventType.Recompute:
                RecomputeCount++;
                TotalRecomputeTimeMs += @event.DurationMs;
                break;
            case CheckpointEventType.Retrieve:
                CacheHitCount++;
                break;
        }
    }
}
```

### Class: ProfilingSummary

```csharp
namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Summary of profiling data
/// </summary>
public class ProfilingSummary
{
    /// <summary>
    /// Start time of profiling
    /// </summary>
    public DateTime StartTime { get; set; }

    /// <summary>
    /// End time of profiling
    /// </summary>
    public DateTime EndTime { get; set; }

    /// <summary>
    /// Total duration in milliseconds
    /// </summary>
    public double Duration { get; set; }

    /// <summary>
    /// Total number of events
    /// </summary>
    public int TotalEvents { get; set; }

    /// <summary>
    /// Total checkpoint time in milliseconds
    /// </summary>
    public long TotalCheckpointTime { get; set; }

    /// <summary>
    /// Total recomputation time in milliseconds
    /// </summary>
    public long TotalRecomputeTime { get; set; }

    /// <summary>
    /// Total memory saved in bytes
    /// </summary>
    public long TotalMemorySaved { get; set; }

    /// <summary>
    /// Layer profiles
    /// </summary>
    public List<LayerProfile> LayerProfiles { get; set; } = new List<LayerProfile>();

    /// <summary>
    /// Creates a string summary
    /// </summary>
    /// <returns>Summary string</returns>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("Checkpoint Profiling Summary");
        sb.AppendLine("============================");
        sb.AppendLine($"Duration: {Duration:F2}ms");
        sb.AppendLine($"Total Events: {TotalEvents}");
        sb.AppendLine($"Total Checkpoint Time: {TotalCheckpointTime}ms");
        sb.AppendLine($"Total Recompute Time: {TotalRecomputeTime}ms");
        sb.AppendLine($"Total Memory Saved: {FormatBytes(TotalMemorySaved)}");
        sb.AppendLine();
        sb.AppendLine("Layer Profiles:");
        sb.AppendLine("===============");
        foreach (var profile in LayerProfiles.OrderByDescending(p => p.TotalMemorySaved))
        {
            sb.AppendLine($"{profile.LayerId}:");
            sb.AppendLine($"  Checkpoints: {profile.CheckpointCount}");
            sb.AppendLine($"  Avg Checkpoint Time: {profile.AverageCheckpointTimeMs:F2}ms");
            sb.AppendLine($"  Recomputations: {profile.RecomputeCount}");
            sb.AppendLine($"  Avg Recompute Time: {profile.AverageRecomputeTimeMs:F2}ms");
            sb.AppendLine($"  Cache Hit Rate: {profile.CacheHitRate:P2}");
            sb.AppendLine($"  Memory Saved: {FormatBytes(profile.TotalMemorySaved)}");
        }

        return sb.ToString();
    }

    private string FormatBytes(long bytes)
    {
        if (bytes < 1024) return $"{bytes}B";
        if (bytes < 1024 * 1024) return $"{bytes / 1024.0:F2}KB";
        if (bytes < 1024 * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F2}MB";
        return $"{bytes / (1024.0 * 1024 * 1024):F2}GB";
    }
}
```

## Optimization Recommendations

### Class: OptimizationRecommender

```csharp
namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Generates optimization recommendations based on profiling data
/// </summary>
public class OptimizationRecommender
{
    /// <summary>
    /// Generates recommendations from profiling summary
    /// </summary>
    /// <param name="summary">Profiling summary</param>
    /// <returns>List of recommendations</returns>
    public List<OptimizationRecommendation> GenerateRecommendations(ProfilingSummary summary)
    {
        var recommendations = new List<OptimizationRecommendation>();

        // Check for layers with high recomputation cost
        var highRecomputeLayers = summary.LayerProfiles
            .Where(p => p.RecomputeCount > p.CheckpointCount * 2)
            .OrderByDescending(p => p.TotalRecomputeTimeMs)
            .Take(3)
            .ToList();

        foreach (var layer in highRecomputeLayers)
        {
            recommendations.Add(new OptimizationRecommendation
            {
                Type = RecommendationType.ReduceRecomputation,
                Priority = RecommendationPriority.High,
                Title = $"Reduce recomputation for layer '{layer.LayerId}'",
                Description = $"Layer '{layer.LayerId}' is recomputed {layer.RecomputeCount} times " +
                              $"but only checkpointed {layer.CheckpointCount} times. " +
                              $"Consider increasing checkpoint frequency or using selective checkpointing.",
                AffectedLayerId = layer.LayerId,
                ExpectedImpact = $"Could save {FormatMs(layer.TotalRecomputeTimeMs)} of recomputation time"
            });
        }

        // Check for layers with low cache hit rate
        var lowCacheHitLayers = summary.LayerProfiles
            .Where(p => p.CheckpointCount > 10 && p.CacheHitRate < 0.5)
            .OrderBy(p => p.CacheHitRate)
            .Take(3)
            .ToList();

        foreach (var layer in lowCacheHitLayers)
        {
            recommendations.Add(new OptimizationRecommendation
            {
                Type = RecommendationType.ImproveCacheHitRate,
                Priority = RecommendationPriority.Medium,
                Title = $"Improve cache hit rate for layer '{layer.LayerId}'",
                Description = $"Layer '{layer.LayerId}' has a cache hit rate of only {layer.CacheHitRate:P0}. " +
                              $"Consider enabling recomputation cache or adjusting checkpoint strategy.",
                AffectedLayerId = layer.LayerId,
                ExpectedImpact = $"Could improve cache hit rate to ~{Math.Min(1.0, layer.CacheHitRate + 0.3):P0}"
            });
        }

        // Check for high memory consumption with low savings
        var lowEfficiencyLayers = summary.LayerProfiles
            .Where(p => p.TotalMemorySaved > 0 &&
                       p.TotalCheckpointTimeMs > p.TotalRecomputeTimeMs * 3)
            .OrderByDescending(p => p.TotalCheckpointTimeMs)
            .Take(3)
            .ToList();

        foreach (var layer in lowEfficiencyLayers)
        {
            recommendations.Add(new OptimizationRecommendation
            {
                Type = RecommendationType.ImproveEfficiency,
                Priority = RecommendationPriority.Low,
                Title = $"Improve checkpointing efficiency for layer '{layer.LayerId}'",
                Description = $"Layer '{layer.LayerId}' has high checkpoint overhead compared to savings. " +
                              $"Consider skipping checkpointing for this layer or using a different strategy.",
                AffectedLayerId = layer.LayerId,
                ExpectedImpact = $"Could reduce overhead by ~{(layer.TotalCheckpointTimeMs - layer.TotalRecomputeTimeMs) * 100 / layer.TotalCheckpointTimeMs:F0}%"
            });
        }

        return recommendations;
    }

    private string FormatMs(long ms)
    {
        if (ms < 1000) return $"{ms}ms";
        if (ms < 60000) return $"{ms / 1000.0:F1}s";
        return $"{ms / 60000.0:F1}m";
    }
}
```

### Class: OptimizationRecommendation

```csharp
namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Optimization recommendation
/// </summary>
public class OptimizationRecommendation
{
    /// <summary>
    /// Type of recommendation
    /// </summary>
    public RecommendationType Type { get; set; }

    /// <summary>
    /// Priority of the recommendation
    /// </summary>
    public RecommendationPriority Priority { get; set; }

    /// <summary>
    /// Title of the recommendation
    /// </summary>
    public string Title { get; set; } = string.Empty;

    /// <summary>
    /// Detailed description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Affected layer ID (if applicable)
    /// </summary>
    public string? AffectedLayerId { get; set; }

    /// <summary>
    /// Expected impact
    /// </summary>
    public string ExpectedImpact { get; set; } = string.Empty;

    /// <summary>
    /// Creates a string representation
    /// </summary>
    /// <returns>String representation</returns>
    public override string ToString()
    {
        return $"[{Priority}] {Title}: {Description}";
    }
}

/// <summary>
/// Type of recommendation
/// </summary>
public enum RecommendationType
{
    ReduceRecomputation,
    ImproveCacheHitRate,
    ImproveEfficiency,
    AdjustCheckpointStrategy,
    EnableAsyncRecomputation,
    IncreaseCheckpointFrequency
}

/// <summary>
/// Priority of recommendation
/// </summary>
public enum RecommendationPriority
{
    High,
    Medium,
    Low
}
```

## Real-time Monitoring

### Class: CheckpointMonitor

```csharp
namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Real-time monitoring for checkpointing operations
/// </summary>
public class CheckpointMonitor : IDisposable
{
    /// <summary>
    /// Event raised when memory usage exceeds threshold
    /// </summary>
    public event EventHandler<MemoryThresholdEventArgs>? MemoryThresholdExceeded;

    /// <summary>
    /// Event raised when recomputation time exceeds threshold
    /// </summary>
    public event EventHandler<RecomputationThresholdEventArgs>? RecomputationThresholdExceeded;

    private readonly CheckpointProfiler _profiler;
    private readonly Timer _monitorTimer;
    private readonly long _memoryThresholdBytes;
    private readonly long _recomputationThresholdMs;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of CheckpointMonitor
    /// </summary>
    /// <param name="profiler">Profiler to monitor</param>
    /// <param name="memoryThresholdBytes">Memory threshold in bytes</param>
    /// <param name="recomputationThresholdMs">Recomputation threshold in milliseconds</param>
    /// <param name="monitorIntervalMs">Monitoring interval in milliseconds</param>
    public CheckpointMonitor(
        CheckpointProfiler profiler,
        long memoryThresholdBytes = 1024 * 1024 * 1024, // 1GB default
        long recomputeThresholdMs = 1000, // 1 second default
        long monitorIntervalMs = 1000) // Check every second
    {
        _profiler = profiler ?? throw new ArgumentNullException(nameof(profiler));
        _memoryThresholdBytes = memoryThresholdBytes;
        _recomputationThresholdMs = recomputeThresholdMs;
        _monitorTimer = new Timer(CheckThresholds, null, monitorIntervalMs, monitorIntervalMs);
        _disposed = false;
    }

    private void CheckThresholds(object? state)
    {
        var summary = _profiler.GetSummary();

        // Check memory threshold
        if (summary.TotalMemorySaved > _memoryThresholdBytes)
        {
            MemoryThresholdExceeded?.Invoke(this, new MemoryThresholdEventArgs
            {
                CurrentMemoryUsage = summary.TotalMemorySaved,
                Threshold = _memoryThresholdBytes,
                Timestamp = DateTime.UtcNow
            });
        }

        // Check recomputation threshold
        if (summary.TotalRecomputeTime > _recomputationThresholdMs)
        {
            RecomputationThresholdExceeded?.Invoke(this, new RecomputationThresholdEventArgs
            {
                CurrentRecomputeTime = summary.TotalRecomputeTime,
                Threshold = _recomputationThresholdMs,
                Timestamp = DateTime.UtcNow
            });
        }
    }

    /// <summary>
    /// Disposes the monitor
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _monitorTimer.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Event arguments for memory threshold exceeded
/// </summary>
public class MemoryThresholdEventArgs : EventArgs
{
    public long CurrentMemoryUsage { get; set; }
    public long Threshold { get; set; }
    public DateTime Timestamp { get; set; }
}

/// <summary>
/// Event arguments for recomputation threshold exceeded
/// </summary>
public class RecomputationThresholdEventArgs : EventArgs
{
    public long CurrentRecomputeTime { get; set; }
    public long Threshold { get; set; }
    public DateTime Timestamp { get; set; }
}
```

## Testing Requirements

### Unit Tests

1. **CheckpointProfiler Tests**
   - [ ] StartProfiling starts profiling
   - [ ] StopProfiling stops profiling
   - [ ] RecordEvent records events correctly
   - [ ] GetLayerProfile returns correct profile
   - [ ] GetSummary returns correct summary
   - [ ] GenerateReport generates correct report
   - [ ] ExportToJson exports correct JSON

2. **LayerProfile Tests**
   - [ ] RecordEvent updates statistics correctly
   - [ ] AverageCheckpointTimeMs calculates correctly
   - [ ] AverageRecomputeTimeMs calculates correctly
   - [ ] CacheHitRate calculates correctly

3. **ProfilingSummary Tests**
   - [ ] ToString generates correct string
   - [ ] FormatBytes formats correctly

4. **OptimizationRecommender Tests**
   - [ ] Generates recommendations for high recomputation layers
   - [ ] Generates recommendations for low cache hit layers
   - [ ] Generates recommendations for low efficiency layers
   - [ ] Prioritizes recommendations correctly

5. **CheckpointMonitor Tests**
   - [ ] Raises MemoryThresholdExceeded event
   - [ ] Raises RecomputationThresholdExceeded event
   - [ ] Checks thresholds at correct intervals

6. **Integration Tests**
   - [ ] End-to-end profiling with checkpointing
   - [ ] Monitoring with real events
   - [ ] Recommendations from actual profiling data

7. **Edge Cases**
   - [ ] Handle profiling with no events
   - [ ] Handle very long profiling sessions
   - [ ] Handle threshold values at boundaries

## Implementation Notes

1. **Performance**:
   - Minimize overhead of profiling
   - Use efficient data structures
   - Consider sampling for high-frequency events

2. **Usability**:
   - Provide clear, actionable recommendations
   - Generate human-readable reports
   - Support both real-time and post-hoc analysis

3. **Extensibility**:
   - Allow custom event types
   - Support custom recommendation logic
   - Enable plugin-based analysis

## Dependencies on Other Specs

This spec depends on:
- **Checkpoint Manager Core** (spec_1) for CheckpointManager
- **Recomputation Engine** (spec_4) for RecomputationEngine

## Estimated Implementation Time
45-60 minutes
