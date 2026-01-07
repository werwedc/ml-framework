using MLFramework.Core;
using Backends = MLFramework.Fusion.Backends;

namespace MLFramework.Fusion.Validation;

/// <summary>
/// Validates thread block configuration constraints
/// </summary>
public class ThreadBlockConstraint : IFusionConstraints
{
    private const int DefaultMaxThreadsPerBlock = 1024;

    private readonly int _maxThreadsPerBlock;

    public ThreadBlockConstraint() : this(DefaultMaxThreadsPerBlock)
    {
    }

    public ThreadBlockConstraint(int maxThreadsPerBlock)
    {
        _maxThreadsPerBlock = maxThreadsPerBlock;
    }

    /// <inheritdoc/>
    public bool Satisfies(IReadOnlyList<Operation> operations)
    {
        return !GetViolations(operations).Any(v => v.Severity == Severity.Error);
    }

    /// <inheritdoc/>
    public IReadOnlyList<ConstraintViolation> GetViolations(IReadOnlyList<Operation> operations)
    {
        var violations = new List<ConstraintViolation>();

        if (operations.Count == 0)
            return violations;

        // Calculate required thread configuration
        var requiredThreads = CalculateRequiredThreads(operations);

        if (requiredThreads > _maxThreadsPerBlock)
        {
            violations.Add(new ConstraintViolation
            {
                ConstraintName = "ThreadBlock",
                Message = $"Requires {requiredThreads} threads, exceeds maximum {_maxThreadsPerBlock}",
                Severity = Severity.Error
            });
        }

        // Check for incompatible thread configurations
        var threadConfigs = operations.Select(op => op.GetThreadBlockConfig()).Where(c => c != null).Distinct().ToList();
        if (threadConfigs.Count > 1)
        {
            violations.Add(new ConstraintViolation
            {
                ConstraintName = "ThreadBlock",
                Message = "Operations have incompatible thread block configurations",
                Severity = Severity.Error
            });
        }

        return violations;
    }

    private int CalculateRequiredThreads(IReadOnlyList<Operation> operations)
    {
        // Estimate based on output tensor size
        var outputShape = operations[^1].OutputShape;
        if (outputShape.Dimensions.Count >= 2)
        {
            return (int)(outputShape.Dimensions[^2] * outputShape.Dimensions[^1]);
        }
        else if (outputShape.Dimensions.Count >= 1)
        {
            return outputShape.Dimensions[^1];
        }
        return 1;
    }
}
