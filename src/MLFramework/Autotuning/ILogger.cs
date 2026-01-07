namespace MLFramework.Autotuning;

/// <summary>
/// Simple logger interface
/// </summary>
public interface ILogger
{
    void LogInformation(string message, params object[] args);
    void LogWarning(string message, params object[] args);
    void LogError(string message, params object[] args);
}
