namespace MLFramework.Fusion;

/// <summary>
/// Interface for managing fusible operations and patterns
/// </summary>
public interface IFusionRegistry
{
    /// <summary>
    /// Registers an operation as fusible with specific constraints
    /// </summary>
    /// <param name="opType">Type of operation to register</param>
    /// <param name="constraints">Constraints for the operation</param>
    void RegisterFusibleOperation(string opType, FusibleOpConstraints constraints);

    /// <summary>
    /// Registers a composite fusion pattern
    /// </summary>
    /// <param name="patternName">Name of the pattern</param>
    /// <param name="pattern">Pattern definition</param>
    void RegisterFusionPattern(string patternName, FusionPatternDefinition pattern);

    /// <summary>
    /// Gets all registered fusible operation types
    /// </summary>
    /// <returns>Set of fusible operation types</returns>
    IReadOnlySet<string> GetFusibleOperations();

    /// <summary>
    /// Gets pattern definition by name
    /// </summary>
    /// <param name="patternName">Name of the pattern</param>
    /// <returns>Pattern definition if found, null otherwise</returns>
    FusionPatternDefinition? GetPattern(string patternName);

    /// <summary>
    /// Finds applicable patterns for a sequence of operations
    /// </summary>
    /// <param name="operations">Operations to find patterns for</param>
    /// <returns>List of pattern matches sorted by priority</returns>
    List<FusionPatternMatch> FindMatches(IEnumerable<Operation> operations);
}
