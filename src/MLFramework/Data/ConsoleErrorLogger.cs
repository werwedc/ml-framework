namespace MLFramework.Data;

/// <summary>
/// Simple console-based error logger implementation.
/// Provides basic logging to the console output.
/// </summary>
public sealed class ConsoleErrorLogger : IErrorLogger
{
    private readonly object _lock = new object();
    private readonly bool _useColors;

    /// <summary>
    /// Initializes a new instance of the ConsoleErrorLogger class.
    /// </summary>
    /// <param name="useColors">Whether to use console colors for different log levels (default: true).</param>
    public ConsoleErrorLogger(bool useColors = true)
    {
        _useColors = useColors && Console.Out != System.IO.StreamWriter.Null;
    }

    /// <summary>
    /// Logs a worker error with full details to the console.
    /// </summary>
    /// <param name="error">The error to log.</param>
    public void LogError(WorkerError error)
    {
        if (error == null)
            throw new ArgumentNullException(nameof(error));

        lock (_lock)
        {
            if (_useColors)
                Console.ForegroundColor = ConsoleColor.Red;

            Console.WriteLine(error.GetFormattedMessage());

            if (!string.IsNullOrEmpty(error.Context))
            {
                Console.WriteLine($"  Context: {error.Context}");
            }

            var stackTrace = error.GetStackTrace();
            if (!string.IsNullOrEmpty(stackTrace))
            {
                Console.WriteLine("  Stack Trace:");
                foreach (var line in stackTrace.Split('\n'))
                {
                    Console.WriteLine($"    {line.Trim()}");
                }
            }

            if (_useColors)
                Console.ResetColor();
        }
    }

    /// <summary>
    /// Logs a warning message to the console.
    /// </summary>
    /// <param name="message">The warning message to log.</param>
    public void LogWarning(string message)
    {
        lock (_lock)
        {
            if (_useColors)
                Console.ForegroundColor = ConsoleColor.Yellow;

            Console.WriteLine($"[WARNING] {message}");

            if (_useColors)
                Console.ResetColor();
        }
    }

    /// <summary>
    /// Logs an informational message to the console.
    /// </summary>
    /// <param name="message">The informational message to log.</param>
    public void LogInfo(string message)
    {
        lock (_lock)
        {
            Console.WriteLine($"[INFO] {message}");
        }
    }

    /// <summary>
    /// Logs a debug message to the console.
    /// </summary>
    /// <param name="message">The debug message to log.</param>
    public void LogDebug(string message)
    {
        lock (_lock)
        {
            if (_useColors)
                Console.ForegroundColor = ConsoleColor.Gray;

            Console.WriteLine($"[DEBUG] {message}");

            if (_useColors)
                Console.ResetColor();
        }
    }
}
