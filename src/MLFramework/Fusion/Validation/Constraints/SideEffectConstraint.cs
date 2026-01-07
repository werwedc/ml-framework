namespace MLFramework.Fusion.Validation;

/// <summary>
/// Validates that operations don't have side effects
/// </summary>
public class SideEffectConstraint : IFusionConstraints
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
            if (HasSideEffects(op))
            {
                violations.Add(new ConstraintViolation
                {
                    ConstraintName = "SideEffect",
                    Message = $"Operation {op.Name} ({op.Type}) has side effects",
                    Severity = Severity.Error
                });
            }
        }

        return violations;
    }

    private bool HasSideEffects(Operation op)
    {
        // Operations with external side effects
        return op.Type is "Print" or "WriteToFile" or "Send" or "Log";
    }
}
