using MLFramework.Fusion.Backends;

namespace MLFramework.Fusion.Profiling;

/// <summary>
/// Interface for logging fusion decisions
/// </summary>
public interface IFusionDecisionLogger
{
    void LogDecision(FusionDecision decision);
    void LogFusionResult(FusedOperation fusedOp, FusionResult result);
    void LogTiming(string message, double durationMs);
}

/// <summary>
/// Console-based fusion decision logger
/// </summary>
public class ConsoleFusionDecisionLogger : IFusionDecisionLogger
{
    private readonly bool _verbose;
    private readonly ILogger _logger;

    public ConsoleFusionDecisionLogger(bool verbose = false, ILogger? logger = null)
    {
        _verbose = verbose;
        _logger = logger ?? new ConsoleLogger();
    }

    public void LogDecision(FusionDecision decision)
    {
        var status = decision.Fused ? "FUSED" : "REJECTED";
        var pattern = decision.PatternType?.ToString() ?? "N/A";
        var reason = decision.RejectionReason ?? "N/A";

        var message = $"[{status}] {decision.OperationChain} (Pattern: {pattern})";

        if (!decision.Fused)
        {
            message += $" Reason: {reason}";
        }

        _logger.LogInformation(message);

        if (_verbose && decision.Metadata.Count > 0)
        {
            _logger.LogInformation("  Metadata:");
            foreach (var (key, value) in decision.Metadata)
            {
                _logger.LogInformation($"    {key}: {value}");
            }
        }
    }

    public void LogFusionResult(FusedOperation fusedOp, FusionResult result)
    {
        _logger.LogInformation($"Fusion Result: {fusedOp.KernelSpec.KernelName}");
        _logger.LogInformation($"  Original ops: {result.OriginalOpCount}");
        _logger.LogInformation($"  Fused ops: {result.FusedOpCount}");
        _logger.LogInformation($"  Fused groups: {result.FusedOperations.Count}");

        if (result.RejectedFusions.Count > 0)
        {
            _logger.LogInformation($"  Rejected: {result.RejectedFusions.Count}");
            foreach (var rejected in result.RejectedFusions)
            {
                _logger.LogInformation($"    - {rejected.RejectionReason}");
            }
        }
    }

    public void LogTiming(string message, double durationMs)
    {
        _logger.LogInformation($"[TIME] {message}: {durationMs:F3}ms");
    }
}

/// <summary>
/// File-based fusion decision logger
/// </summary>
public class FileFusionDecisionLogger : IFusionDecisionLogger
{
    private readonly string _logFilePath;
    private readonly object _lock = new();

    public FileFusionDecisionLogger(string logFilePath)
    {
        _logFilePath = logFilePath;
    }

    public void LogDecision(FusionDecision decision)
    {
        var entry = $"{DateTime.UtcNow:O},{decision.Fused},{decision.OperationChain}," +
                   $"{decision.PatternType},{decision.RejectionReason}\n";

        lock (_lock)
        {
            File.AppendAllText(_logFilePath, entry);
        }
    }

    public void LogFusionResult(FusedOperation fusedOp, FusionResult result)
    {
        var entry = $"{DateTime.UtcNow:O},FUSION_RESULT,{fusedOp.KernelSpec.KernelName}," +
                   $"{result.OriginalOpCount},{result.FusedOpCount},{result.FusedOperations.Count}\n";

        lock (_lock)
        {
            File.AppendAllText(_logFilePath, entry);
        }
    }

    public void LogTiming(string message, double durationMs)
    {
        var entry = $"{DateTime.UtcNow:O},TIMING,{message},{durationMs:F3}\n";

        lock (_lock)
        {
            File.AppendAllText(_logFilePath, entry);
        }
    }
}
