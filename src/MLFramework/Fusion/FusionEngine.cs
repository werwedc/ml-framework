namespace MLFramework.Fusion;

/// <summary>
/// Main fusion engine implementation
/// </summary>
public class FusionEngine : IFusionEngine
{
    private readonly IFusionRegistry _registry;
    private readonly GraphTransformer _transformer;
    private readonly FusionValidator _validator;

    public FusionEngine(IFusionRegistry registry)
    {
        _registry = registry;
        _transformer = new GraphTransformer(registry);
        _validator = new FusionValidator();
    }

    /// <summary>
    /// Applies fusion transformations to a computational graph
    /// </summary>
    public FusionResult ApplyFusion(ComputationalGraph graph, FusionOptions options)
    {
        return _transformer.TransformGraph(graph, options);
    }

    /// <summary>
    /// Fuses a specific set of operations in graph
    /// </summary>
    public FusedOperation FuseOperations(
        IReadOnlyList<Operation> operations,
        FusionPatternDefinition pattern)
    {
        return _transformer.CreateFusedOperation(operations, pattern);
    }

    /// <summary>
    /// Validates that a fusion transformation is correct
    /// </summary>
    public FusionValidationResult ValidateFusion(
        FusedOperation fusedOp,
        ComputationalGraph originalGraph)
    {
        return _validator.ValidateFusion(fusedOp, originalGraph);
    }
}
