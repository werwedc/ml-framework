namespace MLFramework.Fusion.Validation;

/// <summary>
/// Validates that operations have compatible memory layouts
/// </summary>
public class MemoryLayoutConstraint : IFusionConstraints
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

        var referenceLayout = operations[0].Layout;

        for (int i = 1; i < operations.Count; i++)
        {
            var opLayout = operations[i].Layout;

            if (opLayout != referenceLayout && opLayout != TensorLayout.Any && referenceLayout != TensorLayout.Any)
            {
                violations.Add(new ConstraintViolation
                {
                    ConstraintName = "MemoryLayout",
                    Message = $"Operation {i} ({operations[i].Type}) has layout {opLayout}, expected {referenceLayout}",
                    Severity = Severity.Error
                });
            }
        }

        return violations;
    }
}
