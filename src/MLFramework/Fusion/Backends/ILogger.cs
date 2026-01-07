namespace MLFramework.Fusion.Backends;

/// <summary>
/// Simple logger interface for backend operations
/// </summary>
public interface ILogger
{
    /// <summary>
    /// Logs an informational message
    /// </summary>
    void LogInformation(string message, params object?[] args);

    /// <summary>
    /// Logs a warning message
    /// </summary>
    void LogWarning(string message, params object?[] args);

    /// <summary>
    /// Logs an error message
    /// </summary>
    void LogError(string message, params object?[] args);

    /// <summary>
    /// Logs a debug message
    /// </summary>
    void LogDebug(string message, params object?[] args);
}

/// <summary>
/// Console-based logger implementation
/// </summary>
public class ConsoleLogger : ILogger
{
    private readonly bool _debugEnabled;

    public ConsoleLogger(bool debugEnabled = false)
    {
        _debugEnabled = debugEnabled;
    }

    public void LogInformation(string message, params object?[] args)
    {
        Console.WriteLine($"[INFO] {string.Format(message, args)}");
    }

    public void LogWarning(string message, params object?[] args)
    {
        Console.WriteLine($"[WARN] {string.Format(message, args)}");
    }

    public void LogError(string message, params object?[] args)
    {
        Console.Error.WriteLine($"[ERROR] {string.Format(message, args)}");
    }

    public void LogDebug(string message, params object?[] args)
    {
        if (_debugEnabled)
        {
            Console.WriteLine($"[DEBUG] {string.Format(message, args)}");
        }
    }
}
