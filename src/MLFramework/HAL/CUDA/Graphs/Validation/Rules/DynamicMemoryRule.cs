namespace MLFramework.HAL.CUDA.Graphs.Validation.Rules;

/// <summary>
/// Warns about dynamic memory allocations in CUDA graphs
/// </summary>
public class DynamicMemoryRule : IValidationRule
{
    public string RuleName => "DynamicMemoryRule";
    public string Description => "Warns about dynamic memory allocations in graphs";

    public ValidationResult Validate(ICUDAGraph graph)
    {
        var result = new ValidationResult();

        // Note: Full implementation would require tracking memory operations during capture
        // For now, we provide a warning to guide users
        result.Warnings.Add(
            "Ensure all memory is pre-allocated using the graph memory pool to avoid dynamic allocations during graph capture");

        return result;
    }
}
