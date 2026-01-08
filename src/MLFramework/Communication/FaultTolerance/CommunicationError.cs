namespace MLFramework.Communication.FaultTolerance;

/// <summary>
/// Error severity levels
/// </summary>
public enum ErrorSeverity
{
    Warning,
    Recoverable,
    Fatal
}

/// <summary>
/// Error recovery strategy
/// </summary>
public enum RecoveryStrategy
{
    Retry,
    FallbackToDifferentBackend,
    UseDifferentAlgorithm,
    Abort
}

/// <summary>
/// Error information
/// </summary>
public class CommunicationError
{
    public int OperationId { get; set; }
    public string OperationType { get; set; }
    public ErrorSeverity Severity { get; set; }
    public Exception Exception { get; set; }
    public DateTime Timestamp { get; set; }
    public int? Rank { get; set; }
    public Dictionary<string, string> Context { get; set; }

    public override string ToString()
    {
        return $"[{Timestamp:HH:mm:ss.fff}] {OperationType} (Rank: {Rank?.ToString() ?? "N/A"}): " +
               $"{Severity} - {Exception.Message}";
    }
}
