namespace MLFramework.HAL.CUDA.Graphs.Validation.Rules;

/// <summary>
/// Validates that a graph contains at least one operation
/// </summary>
public class EmptyGraphRule : IValidationRule
{
    public string RuleName => "EmptyGraphRule";
    public string Description => "Checks that the graph contains at least one operation";

    public ValidationResult Validate(ICUDAGraph graph)
    {
        var result = new ValidationResult();

        if (graph == null)
        {
            result.Errors.Add("Graph is null");
            return result;
        }

        try
        {
            var validation = graph.Validate();
            if (validation.OperationCount == 0)
            {
                result.Errors.Add("Graph contains no operations to execute");
            }
        }
        catch (Exception ex)
        {
            result.Errors.Add($"Failed to validate graph for operation count: {ex.Message}");
        }

        return result;
    }
}
