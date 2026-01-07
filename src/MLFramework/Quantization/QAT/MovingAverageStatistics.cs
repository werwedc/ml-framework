namespace MLFramework.Quantization.QAT;

/// <summary>
/// Moving average statistics for QAT.
/// </summary>
public class MovingAverageStatistics
{
    private readonly float _momentum;
    private float _mean;
    private float _min;
    private float _max;
    private float _sumSquared;
    private int _count;

    /// <summary>
    /// Gets the current mean value.
    /// </summary>
    public float Mean => _mean;

    /// <summary>
    /// Gets the current minimum value.
    /// </summary>
    public float Min => _min;

    /// <summary>
    /// Gets the current maximum value.
    /// </summary>
    public float Max => _max;

    /// <summary>
    /// Gets the current variance.
    /// </summary>
    public float Variance => _count > 0 ? _sumSquared / _count - _mean * _mean : 0f;

    /// <summary>
    /// Creates moving average statistics.
    /// </summary>
    public MovingAverageStatistics(float momentum = 0.9f)
    {
        _momentum = momentum;
        _mean = 0f;
        _min = float.MaxValue;
        _max = float.MinValue;
        _sumSquared = 0f;
        _count = 0;
    }

    /// <summary>
    /// Update statistics with a single value.
    /// </summary>
    public void Update(float value)
    {
        // Update running statistics
        _mean = _momentum * _mean + (1 - _momentum) * value;
        _min = Math.Min(_min, value);
        _max = Math.Max(_max, value);
        _sumSquared += value * value;
        _count++;
    }

    /// <summary>
    /// Update statistics with a batch of values.
    /// </summary>
    public void UpdateBatch(float[] batch)
    {
        foreach (var value in batch)
        {
            Update(value);
        }
    }

    /// <summary>
    /// Update statistics with a tensor.
    /// </summary>
    public void UpdateTensor(RitterFramework.Core.Tensor.Tensor tensor)
    {
        var data = tensor.Data;
        UpdateBatch(data);
    }

    /// <summary>
    /// Resets all statistics to initial state.
    /// </summary>
    public void Reset()
    {
        _mean = 0f;
        _min = float.MaxValue;
        _max = float.MinValue;
        _sumSquared = 0f;
        _count = 0;
    }

    /// <summary>
    /// Gets all statistics as a summary object.
    /// </summary>
    public Statistics GetStatistics()
    {
        return new Statistics
        {
            Mean = _mean,
            Min = _min,
            Max = _max,
            Variance = Variance,
            Count = _count
        };
    }

    /// <summary>
    /// Statistics summary.
    /// </summary>
    public class Statistics
    {
        public float Mean { get; set; }
        public float Min { get; set; }
        public float Max { get; set; }
        public float Variance { get; set; }
        public int Count { get; set; }
    }
}
