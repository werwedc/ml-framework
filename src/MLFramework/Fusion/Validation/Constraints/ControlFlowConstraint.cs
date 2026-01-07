namespace MLFramework.Fusion.Validation;

/// <summary>
/// Validates control flow constraints (no data-dependent branching or complex control flow)
/// </summary>
public class ControlFlowConstraint : IFusionConstraints
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

        foreach (var op in operations)
        {
            if (HasDataDependentControlFlow(op))
            {
                violations.Add(new ConstraintViolation
                {
                    ConstraintName = "ControlFlow",
                    Message = $"Operation {op.Name} ({op.Type}) has data-dependent control flow",
                    Severity = Severity.Error
                });
            }

            if (HasComplexBranching(op))
            {
                violations.Add(new ConstraintViolation
                {
                    ConstraintName = "ControlFlow",
                    Message = $"Operation {op.Name} ({op.Type}) has complex branching",
                    Severity = Severity.Error
                });
            }
        }

        return violations;
    }

    private bool HasDataDependentControlFlow(Operation op)
    {
        // Operations with dynamic control based on data
        return op.Type == "Where" || op.Type == "DynamicIf" || op.Type == "Conditional";
    }

    private bool HasComplexBranching(Operation op)
    {
        // Operations with complex internal branching
        return op.Type == "Loop" || op.Type == "Recursion" || op.Type == "While";
    }
}
