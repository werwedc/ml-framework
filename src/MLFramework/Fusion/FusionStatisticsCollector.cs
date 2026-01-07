using System.Text;

namespace MLFramework.Fusion;

/// <summary>
/// Collects and tracks fusion statistics
/// </summary>
public class FusionStatisticsCollector : IFusionStatistics
{
    private readonly object _lock = new();
    private int _totalOperations;
    private int _fusedOperations;
    private int _fusedGroups;
    private int _rejectedFusions;
    private readonly Dictionary<FusionPatternType, int> _patternCounts = new();
    private readonly Dictionary<string, int> _rejectionReasons = new();
    private DateTime _startTime = DateTime.UtcNow;
    private DateTime? _endTime;

    /// <summary>
    /// Gets statistics for the current session
    /// </summary>
    public FusionStatistics GetCurrentStatistics()
    {
        lock (_lock)
        {
            return new FusionStatistics
            {
                TotalOperations = _totalOperations,
                FusedOperations = _fusedOperations,
                FusedGroups = _fusedGroups,
                FusionPercentage = _totalOperations > 0
                    ? (_fusedOperations * 100.0 / _totalOperations)
                    : 0.0,
                RejectedFusions = _rejectedFusions,
                AverageOperationsPerFusedGroup = _fusedGroups > 0
                    ? (_fusedOperations * 1.0 / _fusedGroups)
                    : 0.0,
                PatternCounts = new Dictionary<FusionPatternType, int>(_patternCounts),
                RejectionReasons = new Dictionary<string, int>(_rejectionReasons),
                StartTime = _startTime,
                EndTime = _endTime ?? DateTime.UtcNow
            };
        }
    }

    /// <summary>
    /// Resets all statistics
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _totalOperations = 0;
            _fusedOperations = 0;
            _fusedGroups = 0;
            _rejectedFusions = 0;
            _patternCounts.Clear();
            _rejectionReasons.Clear();
            _startTime = DateTime.UtcNow;
            _endTime = null;
        }
    }

    /// <summary>
    /// Logs fusion decisions for debugging
    /// </summary>
    public void LogFusionDecisions()
    {
        var stats = GetCurrentStatistics();
        var sb = new StringBuilder();

        sb.AppendLine("=== Fusion Statistics ===");
        sb.AppendLine($"Total Operations: {stats.TotalOperations}");
        sb.AppendLine($"Fused Operations: {stats.FusedOperations} ({stats.FusionPercentage:F2}%)");
        sb.AppendLine($"Fused Groups: {stats.FusedGroups}");
        sb.AppendLine($"Average Operations per Group: {stats.AverageOperationsPerFusedGroup:F2}");
        sb.AppendLine($"Rejected Fusions: {stats.RejectedFusions}");
        sb.AppendLine();

        sb.AppendLine("Pattern Distribution:");
        foreach (var (pattern, count) in stats.PatternCounts.OrderByDescending(kv => kv.Value))
        {
            sb.AppendLine($"  {pattern}: {count}");
        }

        sb.AppendLine();

        if (stats.RejectionReasons.Count > 0)
        {
            sb.AppendLine("Rejection Reasons:");
            foreach (var (reason, count) in stats.RejectionReasons.OrderByDescending(kv => kv.Value))
            {
                sb.AppendLine($"  {reason}: {count}");
            }
        }

        sb.AppendLine($"Duration: {(stats.EndTime - stats.StartTime).TotalSeconds:F2}s");

        Console.WriteLine(sb.ToString());
    }

    /// <summary>
    /// Records a single operation
    /// </summary>
    public void RecordOperation(Operation op)
    {
        lock (_lock)
        {
            _totalOperations++;
        }
    }

    /// <summary>
    /// Records a successfully fused group of operations
    /// </summary>
    public void RecordFusedGroup(IReadOnlyList<Operation> operations, FusionPatternType patternType)
    {
        lock (_lock)
        {
            _fusedOperations += operations.Count;
            _fusedGroups++;
            _patternCounts.TryGetValue(patternType, out var count);
            _patternCounts[patternType] = count + 1;
        }
    }

    /// <summary>
    /// Records a rejected fusion attempt
    /// </summary>
    public void RecordRejection(string reason)
    {
        lock (_lock)
        {
            _rejectedFusions++;
            _rejectionReasons.TryGetValue(reason, out var count);
            _rejectionReasons[reason] = count + 1;
        }
    }
}
