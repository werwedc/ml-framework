using System.Collections.Concurrent;

namespace MLFramework.Visualization.Profiling.Statistics;

/// <summary>
/// Tracks duration statistics for profiling operations
/// </summary>
public class DurationTracker
{
    private readonly ConcurrentBag<long> _durations;
    private readonly object _statisticsLock = new();
    private long _minDuration;
    private long _maxDuration;
    private double _sum;
    private double _sumOfSquares;
    private int _count;

    /// <summary>
    /// Gets the minimum duration in nanoseconds
    /// </summary>
    public long MinDurationNanoseconds => _count > 0 ? _minDuration : 0;

    /// <summary>
    /// Gets the maximum duration in nanoseconds
    /// </summary>
    public long MaxDurationNanoseconds => _count > 0 ? _maxDuration : 0;

    /// <summary>
    /// Gets the count of durations
    /// </summary>
    public int Count => _count;

    /// <summary>
    /// Gets the sum of all durations in nanoseconds
    /// </summary>
    public long TotalDurationNanoseconds => (long)_sum;

    /// <summary>
    /// Gets the average duration in nanoseconds
    /// </summary>
    public double AverageDurationNanoseconds => _count > 0 ? _sum / _count : 0.0;

    /// <summary>
    /// Gets the standard deviation in nanoseconds
    /// </summary>
    public double StdDevNanoseconds => _count > 1 ? Math.Sqrt(Math.Max(0, (_sumOfSquares - _sum * _sum / _count) / (_count - 1))) : 0.0;

    /// <summary>
    /// Creates a new duration tracker
    /// </summary>
    public DurationTracker()
    {
        _durations = new ConcurrentBag<long>();
        _minDuration = long.MaxValue;
        _maxDuration = long.MinValue;
    }

    /// <summary>
    /// Records a duration
    /// </summary>
    /// <param name="durationNanoseconds">Duration in nanoseconds</param>
    public void RecordDuration(long durationNanoseconds)
    {
        _durations.Add(durationNanoseconds);

        lock (_statisticsLock)
        {
            _count++;
            _sum += durationNanoseconds;
            _sumOfSquares += (double)durationNanoseconds * durationNanoseconds;

            if (durationNanoseconds < _minDuration)
            {
                _minDuration = durationNanoseconds;
            }

            if (durationNanoseconds > _maxDuration)
            {
                _maxDuration = durationNanoseconds;
            }
        }
    }

    /// <summary>
    /// Gets all recorded durations as an array
    /// </summary>
    /// <returns>Array of durations in nanoseconds</returns>
    public long[] GetDurations()
    {
        return _durations.ToArray();
    }

    /// <summary>
    /// Clears all recorded durations
    /// </summary>
    public void Clear()
    {
        _durations.Clear();

        lock (_statisticsLock)
        {
            _count = 0;
            _sum = 0;
            _sumOfSquares = 0;
            _minDuration = long.MaxValue;
            _maxDuration = long.MinValue;
        }
    }
}
