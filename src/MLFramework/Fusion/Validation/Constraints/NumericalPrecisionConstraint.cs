using MLFramework.Core;

namespace MLFramework.Fusion.Validation;

/// <summary>
/// Validates that operations have compatible numerical precision
/// </summary>
public class NumericalPrecisionConstraint : IFusionConstraints
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

        var referenceDtype = operations[0].DataType;

        for (int i = 1; i < operations.Count; i++)
        {
            var opDtype = operations[i].DataType;

            if (opDtype != referenceDtype)
            {
                violations.Add(new ConstraintViolation
                {
                    ConstraintName = "NumericalPrecision",
                    Message = $"Operation {i} ({operations[i].Type}) has dtype {opDtype}, expected {referenceDtype}",
                    Severity = Severity.Error
                });
            }
        }

        // Check for precision-sensitive operations
        var hasPrecisionSensitiveOps = operations.Any(op =>
            op.Type == "ReduceSum" || op.Type == "ReduceMean");

        if (hasPrecisionSensitiveOps && referenceDtype == DataType.Float16)
        {
            violations.Add(new ConstraintViolation
            {
                ConstraintName = "NumericalPrecision",
                Message = "Precision-sensitive operations with FP16 may cause numerical instability",
                Severity = Severity.Warning
            });
        }

        return violations;
    }
}
