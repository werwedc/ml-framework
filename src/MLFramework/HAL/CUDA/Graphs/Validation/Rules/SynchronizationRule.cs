namespace MLFramework.HAL.CUDA.Graphs.Validation.Rules;

/// <summary>
/// Warns about CPU-GPU synchronization in CUDA graphs
/// </summary>
public class SynchronizationRule : IValidationRule
{
    public string RuleName => "SynchronizationRule";
    public string Description => "Warns about CPU-GPU synchronization in graphs";

    public ValidationResult Validate(ICUDAGraph graph)
    {
        var result = new ValidationResult();

        // Note: Full implementation would require analyzing captured operations for sync points
        // For now, we provide a warning to guide users
        result.Warnings.Add(
            "Avoid CPU-GPU synchronization points (e.g., device synchronize, event synchronization) within graph capture as it can break graph execution");

        return result;
    }
}
