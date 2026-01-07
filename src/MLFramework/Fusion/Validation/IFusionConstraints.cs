namespace MLFramework.Fusion.Validation;

/// <summary>
/// Interface for fusion constraints validators
/// </summary>
public interface IFusionConstraints
{
    /// <summary>
    /// Validates that operations satisfy fusion constraints
    /// </summary>
    bool Satisfies(IReadOnlyList<Operation> operations);

    /// <summary>
    /// Gets detailed constraint violations
    /// </summary>
    IReadOnlyList<ConstraintViolation> GetViolations(IReadOnlyList<Operation> operations);
}
