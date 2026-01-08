namespace MLFramework.Visualization;

/// <summary>
/// Interface for visualization backends that can log metrics and profiles.
/// This provides a contract for different visualization tools (TensorBoard, Weights & Biases, etc.)
/// </summary>
public interface IVisualizer
{
    /// <summary>
    /// Logs a scalar metric value
    /// </summary>
    /// <param name="name">Metric name (e.g., "train/loss")</param>
    /// <param name="value">Metric value</param>
    /// <param name="step">Training step number</param>
    void LogScalar(string name, float value, long step);

    /// <summary>
    /// Starts profiling a section of code
    /// </summary>
    /// <param name="name">Name of the section being profiled</param>
    void StartProfile(string name);

    /// <summary>
    /// Ends profiling and records the elapsed time
    /// </summary>
    /// <param name="name">Name of the section being profiled</param>
    void EndProfile(string name);

    /// <summary>
    /// Flushes any pending logs
    /// </summary>
    void Flush();
}

/// <summary>
/// Simple console-based visualizer for testing and development.
/// Outputs logs to the console instead of a file.
/// </summary>
public class ConsoleVisualizer : IVisualizer
{
    private readonly Dictionary<string, DateTime> _profileStartTimes = new();

    public void LogScalar(string name, float value, long step)
    {
        Console.WriteLine($"[Step {step}] {name}: {value:F4}");
    }

    public void StartProfile(string name)
    {
        _profileStartTimes[name] = DateTime.UtcNow;
    }

    public void EndProfile(string name)
    {
        if (_profileStartTimes.TryGetValue(name, out DateTime startTime))
        {
            var elapsed = DateTime.UtcNow - startTime;
            Console.WriteLine($"[Profile] {name}: {elapsed.TotalMilliseconds:F2}ms");
            _profileStartTimes.Remove(name);
        }
    }

    public void Flush()
    {
        Console.Out.Flush();
    }
}

/// <summary>
/// TensorBoard-compatible visualizer.
/// Logs metrics to TensorBoard event files.
/// </summary>
public class TensorBoardVisualizer : IVisualizer
{
    private readonly string _logDirectory;
    private readonly object _lock = new();
    private readonly Dictionary<string, DateTime> _profileStartTimes = new();

    /// <summary>
    /// Creates a new TensorBoard visualizer
    /// </summary>
    /// <param name="logDirectory">Directory where logs will be written</param>
    public TensorBoardVisualizer(string logDirectory)
    {
        _logDirectory = logDirectory ?? throw new ArgumentNullException(nameof(logDirectory));

        // Ensure log directory exists
        if (!Directory.Exists(_logDirectory))
        {
            Directory.CreateDirectory(_logDirectory);
        }
    }

    public void LogScalar(string name, float value, long step)
    {
        lock (_lock)
        {
            Console.WriteLine($"[TensorBoard] {name} = {value} (step {step})");
            // In a real implementation, this would write to TensorBoard event files
            // For now, we'll just log to console
        }
    }

    public void StartProfile(string name)
    {
        lock (_lock)
        {
            _profileStartTimes[name] = DateTime.UtcNow;
        }
    }

    public void EndProfile(string name)
    {
        lock (_lock)
        {
            if (_profileStartTimes.TryGetValue(name, out DateTime startTime))
            {
                var elapsed = DateTime.UtcNow - startTime;
                Console.WriteLine($"[TensorBoard Profile] {name}: {elapsed.TotalMilliseconds:F2}ms");
                _profileStartTimes.Remove(name);
                // In a real implementation, this would write to TensorBoard event files
            }
        }
    }

    public void Flush()
    {
        lock (_lock)
        {
            // Flush any pending writes
            Console.Out.Flush();
        }
    }
}
