using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.QAT;

/// <summary>
/// Observer strategy for collecting activation statistics.
/// </summary>
public enum ObserverStrategy
{
    /// <summary>
    /// Track absolute min and max values.
    /// </summary>
    MinMax,

    /// <summary>
    /// Use moving average for statistics.
    /// </summary>
    MovingAverage,

    /// <summary>
    /// Use entropy-based calibration.
    /// </summary>
    Entropy
}

/// <summary>
/// Observer statistics.
/// </summary>
public class ObserverStatistics
{
    public float Min { get; set; }
    public float Max { get; set; }
    public float Mean { get; set; }
    public float Variance { get; set; }
    public int Count { get; set; }
}

/// <summary>
/// Activation observer for collecting statistics during QAT training.
/// </summary>
public class ActivationObserver
{
    private readonly ObserverStrategy _strategy;
    private readonly float _momentum;
    private readonly bool _perChannel;
    private float _min;
    private float _max;
    private float _mean;
    private float _sumSquared;
    private int _count;
    private bool _enabled;

    /// <summary>
    /// Gets or sets whether the observer is enabled.
    /// </summary>
    public bool Enabled
    {
        get => _enabled;
        set => _enabled = value;
    }

    /// <summary>
    /// Creates an activation observer.
    /// </summary>
    public ActivationObserver(ObserverStrategy strategy = ObserverStrategy.MinMax, float momentum = 0.9f, bool perChannel = false)
    {
        _strategy = strategy;
        _momentum = momentum;
        _perChannel = perChannel;
        _min = float.MaxValue;
        _max = float.MinValue;
        _mean = 0f;
        _sumSquared = 0f;
        _count = 0;
        _enabled = true;
    }

    /// <summary>
    /// Update observer statistics with a tensor.
    /// </summary>
    public void Update(RitterFramework.Core.Tensor.Tensor tensor)
    {
        if (!_enabled) return;

        var data = tensor.Data;

        if (_strategy == ObserverStrategy.MinMax)
        {
            foreach (var value in data)
            {
                _min = Math.Min(_min, value);
                _max = Math.Max(_max, value);
                _mean = (_mean * _count + value) / (_count + 1);
                _sumSquared += value * value;
                _count++;
            }
        }
        else if (_strategy == ObserverStrategy.MovingAverage)
        {
            foreach (var value in data)
            {
                _min = Math.Min(_min, value);
                _max = Math.Max(_max, value);
                _mean = _momentum * _mean + (1 - _momentum) * value;
                _sumSquared += value * value;
                _count++;
            }
        }
        else if (_strategy == ObserverStrategy.Entropy)
        {
            // Similar to MinMax for now
            foreach (var value in data)
            {
                _min = Math.Min(_min, value);
                _max = Math.Max(_max, value);
                _mean = (_mean * _count + value) / (_count + 1);
                _sumSquared += value * value;
                _count++;
            }
        }
    }

    /// <summary>
    /// Gets the collected statistics.
    /// </summary>
    public ObserverStatistics GetStatistics()
    {
        float variance = _count > 0 ? _sumSquared / _count - _mean * _mean : 0f;
        return new ObserverStatistics
        {
            Min = _min == float.MaxValue ? 0f : _min,
            Max = _max == float.MinValue ? 0f : _max,
            Mean = _mean,
            Variance = variance,
            Count = _count
        };
    }

    /// <summary>
    /// Gets quantization parameters based on collected statistics.
    /// </summary>
    public QuantizationParameters? GetQuantizationParameters()
    {
        var stats = GetStatistics();
        float range = stats.Max - stats.Min;

        if (range < 1e-6f)
        {
            // Handle zero range case
            return new QuantizationParameters
            {
                Scale = 1.0f,
                ZeroPoint = 0,
                Mode = QuantizationMode.PerTensorSymmetric
            };
        }

        // Calculate scale for Int8 range [-128, 127]
        float scale = range / 255f;
        int zeroPoint = (int)(-stats.Min / scale);

        return new QuantizationParameters
        {
            Scale = scale,
            ZeroPoint = zeroPoint,
            Mode = QuantizationMode.PerTensorAsymmetric
        };
    }

    /// <summary>
    /// Resets the observer statistics.
    /// </summary>
    public void Reset()
    {
        _min = float.MaxValue;
        _max = float.MinValue;
        _mean = 0f;
        _sumSquared = 0f;
        _count = 0;
    }
}
