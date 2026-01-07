namespace MLFramework.Fusion;

/// <summary>
/// Interface for shared memory planning
/// </summary>
public interface ISharedMemoryPlanner
{
    /// <summary>
    /// Plans memory layout for shared memory optimization
    /// </summary>
    MemoryLayout PlanMemory(FusionIR ir, GenerationOptions options);
}
