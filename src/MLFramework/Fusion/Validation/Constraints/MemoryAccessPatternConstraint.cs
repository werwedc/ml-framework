namespace MLFramework.Fusion.Validation;

/// <summary>
/// Validates memory access pattern compatibility
/// </summary>
public class MemoryAccessPatternConstraint : IFusionConstraints
{
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

        var analyzer = new GraphAnalyzer();
        var patterns = operations.Select(op => analyzer.AnalyzeAccessPattern(op)).ToList();

        // Check for incompatible access patterns
        for (int i = 1; i < patterns.Count; i++)
        {
            if (!ArePatternsCompatible(patterns[i - 1], patterns[i]))
            {
                violations.Add(new ConstraintViolation
                {
                    ConstraintName = "MemoryAccessPattern",
                    Message = $"Incompatible access patterns: {patterns[i-1]} and {patterns[i]}",
                    Severity = Severity.Error
                });
            }
        }

        // Check for gather/scatter operations (hard to fuse)
        if (patterns.Any(p => p == MemoryAccessPattern.Gather || p == MemoryAccessPattern.Scatter))
        {
            violations.Add(new ConstraintViolation
            {
                ConstraintName = "MemoryAccessPattern",
                Message = "Gather/Scatter operations are difficult to fuse efficiently",
                Severity = Severity.Warning
            });
        }

        return violations;
    }

    private bool ArePatternsCompatible(MemoryAccessPattern p1, MemoryAccessPattern p2)
    {
        // Define compatible pattern combinations
        var compatiblePairs = new HashSet<(MemoryAccessPattern, MemoryAccessPattern)>
        {
            (MemoryAccessPattern.ElementWise, MemoryAccessPattern.ElementWise),
            (MemoryAccessPattern.ElementWise, MemoryAccessPattern.Reduction),
            (MemoryAccessPattern.Spatial, MemoryAccessPattern.ElementWise)
        };

        return compatiblePairs.Contains((p1, p2)) || compatiblePairs.Contains((p2, p1));
    }
}
