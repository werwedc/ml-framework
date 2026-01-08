using MachineLearning.Visualization.Events;

namespace MachineLearning.Visualization.Tests;

/// <summary>
/// Simple test event type
/// </summary>
public class TestEvent : Event
{
    public string Message { get; set; } = string.Empty;
    public int Value { get; set; }

    public TestEvent()
    {
    }

    public TestEvent(string message, int value)
    {
        Message = message;
        Value = value;
    }

    public override byte[] Serialize()
    {
        return System.Text.Encoding.UTF8.GetBytes($"{Message}|{Value}");
    }

    public override void Deserialize(byte[] data)
    {
        var str = System.Text.Encoding.UTF8.GetString(data);
        var parts = str.Split('|');
        Message = parts[0];
        Value = int.Parse(parts[1]);
    }
}

/// <summary>
/// Another test event type for testing type-based routing
/// </summary>
public class MetricEvent : Event
{
    public string MetricName { get; set; } = string.Empty;
    public double MetricValue { get; set; }

    public MetricEvent()
    {
    }

    public MetricEvent(string metricName, double metricValue)
    {
        MetricName = metricName;
        MetricValue = metricValue;
    }

    public override byte[] Serialize()
    {
        return System.Text.Encoding.UTF8.GetBytes($"{MetricName}|{MetricValue}");
    }

    public override void Deserialize(byte[] data)
    {
        var str = System.Text.Encoding.UTF8.GetString(data);
        var parts = str.Split('|');
        MetricName = parts[0];
        MetricValue = double.Parse(parts[1]);
    }
}
