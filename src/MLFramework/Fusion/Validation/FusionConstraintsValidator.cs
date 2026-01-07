using MLFramework.Core;

namespace MLFramework.Fusion.Validation;

/// <summary>
/// Validates fusion constraints across multiple constraint types
/// </summary>
public class FusionConstraintsValidator
{
    private readonly List<IFusionConstraints> _constraints;

    /// <summary>
    /// Creates a new FusionConstraintsValidator with default constraints
    /// </summary>
    public FusionConstraintsValidator()
    {
        _constraints = new List<IFusionConstraints>
        {
            new MemoryLayoutConstraint(),
            new NumericalPrecisionConstraint(),
            new ThreadBlockConstraint(),
            new SideEffectConstraint(),
            new ControlFlowConstraint(),
            new MemoryAccessPatternConstraint()
        };
    }

    /// <summary>
    /// Creates a new FusionConstraintsValidator with custom constraints
    /// </summary>
    public FusionConstraintsValidator(IEnumerable<IFusionConstraints> constraints)
    {
        _constraints = constraints.ToList();
    }

    /// <summary>
    /// Validates operations against all constraints
    /// </summary>
    /// <param name="operations">Operations to validate</param>
    /// <param name="violations">Output parameter for constraint violations</param>
    /// <returns>True if all operations satisfy all constraints, false otherwise</returns>
    public bool Validate(IReadOnlyList<Operation> operations, out IReadOnlyList<ConstraintViolation> violations)
    {
        var allViolations = new List<ConstraintViolation>();

        foreach (var constraint in _constraints)
        {
            var constraintViolations = constraint.GetViolations(operations);
            allViolations.AddRange(constraintViolations);
        }

        violations = allViolations;
        return !violations.Any(v => v.Severity == Severity.Error);
    }

    /// <summary>
    /// Gets all constraint violations for the given operations
    /// </summary>
    public IReadOnlyList<ConstraintViolation> GetAllViolations(IReadOnlyList<Operation> operations)
    {
        var allViolations = new List<ConstraintViolation>();

        foreach (var constraint in _constraints)
        {
            allViolations.AddRange(constraint.GetViolations(operations));
        }

        return allViolations;
    }

    /// <summary>
    /// Checks if operations can be fused (only errors, not warnings)
    /// </summary>
    public bool CanFuse(IReadOnlyList<Operation> operations)
    {
        Validate(operations, out var violations);
        return !violations.Any(v => v.Severity == Severity.Error);
    }

    /// <summary>
    /// Gets only error-level violations
    /// </summary>
    public IReadOnlyList<ConstraintViolation> GetErrorViolations(IReadOnlyList<Operation> operations)
    {
        return GetAllViolations(operations)
            .Where(v => v.Severity == Severity.Error)
            .ToList();
    }

    /// <summary>
    /// Gets only warning-level violations
    /// </summary>
    public IReadOnlyList<ConstraintViolation> GetWarningViolations(IReadOnlyList<Operation> operations)
    {
        return GetAllViolations(operations)
            .Where(v => v.Severity == Severity.Warning)
            .ToList();
    }
}
