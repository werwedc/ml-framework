namespace MLFramework.HAL.CUDA.Graphs.Validation.Rules;

/// <summary>
/// Warns about data-dependent control flow in CUDA graphs
/// </summary>
public class ControlFlowRule : IValidationRule
{
    public string RuleName => "ControlFlowRule";
    public string Description => "Warns about data-dependent control flow in graphs";

    public ValidationResult Validate(ICUDAGraph graph)
    {
        var result = new ValidationResult();

        // Note: Full implementation would require analyzing captured operations for branching
        // For now, we provide a warning to guide users
        result.Warnings.Add(
            "Avoid data-dependent control flow (e.g., if statements based on GPU values) within graph capture as it can cause undefined behavior");

        return result;
    }
}
