namespace MLFramework.Fusion.Validation;

/// <summary>
/// Represents a constraint violation
/// </summary>
public record ConstraintViolation
{
    /// <summary>
    /// Name of the violated constraint
    /// </summary>
    public required string ConstraintName { get; init; }

    /// <summary>
    /// Detailed message about the violation
    /// </summary>
    public required string Message { get; init; }

    /// <summary>
    /// Severity level of the violation
    /// </summary>
    public required Severity Severity { get; init; }

    public override string ToString()
    {
        return $"[{Severity}] {ConstraintName}: {Message}";
    }
}
