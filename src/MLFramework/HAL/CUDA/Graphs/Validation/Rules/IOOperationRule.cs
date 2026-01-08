namespace MLFramework.HAL.CUDA.Graphs.Validation.Rules;

/// <summary>
/// Warns about I/O operations in CUDA graphs
/// </summary>
public class IOOperationRule : IValidationRule
{
    public string RuleName => "IOOperationRule";
    public string Description => "Warns about I/O operations in graphs";

    public ValidationResult Validate(ICUDAGraph graph)
    {
        var result = new ValidationResult();

        // Note: Full implementation would require analyzing captured operations for I/O
        // For now, we provide a warning to guide users
        result.Warnings.Add(
            "Avoid I/O operations (e.g., file operations, console output, network calls) within graph capture as they are not supported");

        return result;
    }
}
