using MLFramework.Fusion.Backends;

namespace MLFramework.Fusion;

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
    public required IReadOnlyList<KernelLaunchParameter> Parameters { get; init; }
}
