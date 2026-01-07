using MLFramework.Core;

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

/// <summary>
/// Detailed specification for kernel compilation and execution
/// </summary>
public record KernelSpecification
{
    public required string KernelName { get; init; }
    public required FusionStrategy Strategy { get; init; }
    public required IReadOnlyList<FusionVariable> InputTensors { get; init; }
    public required IReadOnlyList<FusionVariable> OutputTensors { get; init; }
    public required int TemporaryMemoryBytes { get; init; }
    public required int RegisterBytes { get; init; }
    public required ThreadBlockConfiguration ThreadBlockConfig { get; init; }
    public required IReadOnlyList<string> CompilationFlags { get; init; }
}

/// <summary>
/// Kernel launch configuration
/// </summary>
public record KernelLaunchConfiguration
{
    public required ThreadBlockConfiguration BlockDim { get; init; }
    public required ThreadBlockConfiguration GridDim { get; init; }
    public required int SharedMemoryBytes { get; init; }
    public required IReadOnlyList<KernelLaunchParameter> Parameters { get; init; }
}

/// <summary>
/// Parameter for kernel launch
/// </summary>
public record KernelLaunchParameter
{
    public required string Name { get; init; }
    public required object? Value { get; init; }
    public required KernelParameterType Type { get; init; }
}

/// <summary>
/// Type of kernel parameter
/// </summary>
public enum KernelParameterType
{
    Tensor,
    Scalar,
    Pointer,
    Int,
    Float
}
