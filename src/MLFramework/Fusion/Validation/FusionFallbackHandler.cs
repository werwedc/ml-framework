using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;
using MLFramework.Autotuning;

namespace MLFramework.Fusion.Validation;

/// <summary>
/// Handles execution fallback when fusion cannot be applied
/// </summary>
public class FusionFallbackHandler : IFusionFallback
{
    private readonly ILogger _logger;
    private readonly IKernelExecutor _executor;

    /// <summary>
    /// Creates a new FusionFallbackHandler
    /// </summary>
    public FusionFallbackHandler(ILogger logger, IKernelExecutor executor)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
    }

    /// <inheritdoc/>
    public Tensor ExecuteSeparate(IReadOnlyList<Operation> operations, Tensor input)
    {
        if (operations == null || operations.Count == 0)
        {
            throw new ArgumentException("Operations list cannot be null or empty", nameof(operations));
        }

        Tensor output = input;

        foreach (var op in operations)
        {
            output = _executor.ExecuteKernel(op, output);
        }

        _executor.Synchronize();
        return output;
    }

    /// <inheritdoc/>
    public void LogFallbackReason(string reason, IReadOnlyList<Operation> operations)
    {
        if (operations == null || operations.Count == 0)
        {
            _logger.LogWarning("Fusion fallback: {Reason}", reason);
            return;
        }

        var opTypes = string.Join(" -> ", operations.Select(op => op.Type));
        _logger.LogWarning(
            "Fusion fallback for chain: {OpChain}. Reason: {Reason}",
            opTypes, reason);
    }

    /// <summary>
    /// Executes operations with fallback if fusion fails
    /// </summary>
    /// <param name="operations">Operations to execute</param>
    /// <param name="input">Input tensor</param>
    /// <param name="violations">Constraint violations that caused the fallback</param>
    /// <returns>Output tensor</returns>
    public Tensor ExecuteWithFallback(
        IReadOnlyList<Operation> operations,
        Tensor input,
        IReadOnlyList<ConstraintViolation> violations)
    {
        if (violations == null || violations.Count == 0)
        {
            return ExecuteSeparate(operations, input);
        }

        var errorViolations = violations.Where(v => v.Severity == Severity.Error).ToList();
        var reason = string.Join("; ", errorViolations.Select(v => v.Message));

        LogFallbackReason(reason, operations);
        return ExecuteSeparate(operations, input);
    }
}
