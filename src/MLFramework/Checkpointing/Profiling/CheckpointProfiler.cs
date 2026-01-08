using System.Text.Json;

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
