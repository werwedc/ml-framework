using MLFramework.Core;
using MLFramework.Fusion.Backends;

namespace MLFramework.Fusion;

/// <summary>
/// Represents a fused operation in computational graph
/// </summary>
public record FusedOperation : Operation
{
    public required IReadOnlyList<Operation> ConstituentOperations { get; init; }
    public required FusionPatternDefinition Pattern { get; init; }
    public required FusionIR IntermediateRepresentation { get; init; }
    public required KernelSpecification KernelSpec { get; init; }

    public string ComputedName => $"Fused_{Pattern.Name}_{Id}";

    public static FusedOperation Create(
        string id,
        IReadOnlyList<Operation> operations,
        FusionPatternDefinition pattern,
        FusionIR ir,
        KernelSpecification kernelSpec)
    {
        return new FusedOperation
        {
            Id = id,
            Type = $"Fused_{pattern.Name}",
            Name = $"Fused_{pattern.Name}_{id}",
            ConstituentOperations = operations,
            Pattern = pattern,
            IntermediateRepresentation = ir,
            KernelSpec = kernelSpec,
            InputShape = operations[0].InputShape,
            OutputShape = operations[^1].OutputShape,
            DataType = operations[0].DataType,
            Layout = operations[0].Layout,
            Inputs = operations[0].Inputs,
            Outputs = operations[^1].Outputs,
            Attributes = new Dictionary<string, object>()
        };
    }
}
