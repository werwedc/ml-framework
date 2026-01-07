namespace MLFramework.Fusion;

/// <summary>
/// Core interface for performing graph fusion
/// </summary>
public interface IFusionEngine
{
    /// <summary>
    /// Applies fusion transformations to a computational graph
    /// </summary>
    FusionResult ApplyFusion(ComputationalGraph graph, FusionOptions options);

    /// <summary>
    /// Fuses a specific set of operations in graph
    /// </summary>
    FusedOperation FuseOperations(IReadOnlyList<Operation> operations, FusionPatternDefinition pattern);

    /// <summary>
    /// Validates that a fusion transformation is correct
    /// </summary>
    FusionValidationResult ValidateFusion(FusedOperation fusedOp, ComputationalGraph originalGraph);
}
