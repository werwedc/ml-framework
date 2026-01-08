using System.Reflection;
using MLFramework.HAL.CUDA.Graphs.Validation.Rules;

namespace MLFramework.HAL.CUDA.Graphs.Validation;

/// <summary>
/// Comprehensive validation system for CUDA graphs that checks operations
/// for graph compatibility and provides detailed feedback on potential issues
/// </summary>
public class CUDAGraphValidator
{
    private readonly List<IValidationRule> _rules;

    public CUDAGraphValidator()
    {
        _rules = new List<IValidationRule>();
        RegisterDefaultRules();
    }

    /// <summary>
    /// Validates a captured graph using all registered validation rules
    /// </summary>
    /// <param name="graph">The graph to validate</param>
    /// <returns>Validation result with errors, warnings, and operation count</returns>
    public CUDAGraphValidationResult Validate(ICUDAGraph graph)
    {
        if (graph == null)
        {
            return CUDAGraphValidationResult.Failure(
                new[] { "Graph is null" }, 0);
        }

        var errors = new List<string>();
        var warnings = new List<string>();

        // Get operation count
        var operationCount = GetOperationCount(graph);

        if (operationCount == 0)
        {
            errors.Add("Graph contains no operations");
        }

        // Run all validation rules
        foreach (var rule in _rules)
        {
            try
            {
                var result = rule.Validate(graph);
                errors.AddRange(result.Errors);
                warnings.AddRange(result.Warnings);
            }
            catch (Exception ex)
            {
                errors.Add($"Validation rule '{rule.RuleName}' threw an exception: {ex.Message}");
            }
        }

        return new CUDAGraphValidationResult
        {
            IsValid = errors.Count == 0,
            Errors = errors,
            Warnings = warnings,
            OperationCount = operationCount
        };
    }

    /// <summary>
    /// Validates a graph instance by attempting to instantiate it
    /// </summary>
    /// <param name="graphHandle">Native handle to the CUDA graph</param>
    /// <returns>Validation result with errors and operation count</returns>
    public CUDAGraphValidationResult ValidateGraphInstance(IntPtr graphHandle)
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        // Check if graph handle is valid
        if (graphHandle == IntPtr.Zero)
        {
            errors.Add("Graph handle is null");
            return new CUDAGraphValidationResult
            {
                IsValid = false,
                Errors = errors,
                Warnings = warnings,
                OperationCount = 0
            };
        }

        // Try to instantiate the graph
        var result = CudaApi.CudaGraphInstantiate(
            out IntPtr graphExec,
            graphHandle,
            IntPtr.Zero,
            0);

        if (result != CudaError.Success)
        {
            errors.Add($"Graph instantiation failed: {GetErrorDescription(result)}");
        }
        else
        {
            // Clean up the instantiated graph
            CudaApi.CudaGraphExecDestroy(graphExec);
        }

        // Get node count
        var nodeResult = CudaApi.CudaGraphGetNodes(graphHandle, out _, out ulong nodeCount);
        if (nodeResult != CudaError.Success)
        {
            errors.Add("Failed to get graph nodes");
        }

        return new CUDAGraphValidationResult
        {
            IsValid = errors.Count == 0,
            Errors = errors,
            Warnings = warnings,
            OperationCount = (int)nodeCount
        };
    }

    /// <summary>
    /// Registers a custom validation rule
    /// </summary>
    /// <param name="rule">The validation rule to register</param>
    public void RegisterRule(IValidationRule rule)
    {
        if (rule == null)
            throw new ArgumentNullException(nameof(rule));

        _rules.Add(rule);
    }

    /// <summary>
    /// Clears all validation rules
    /// </summary>
    public void ClearRules()
    {
        _rules.Clear();
    }

    /// <summary>
    /// Gets all registered validation rules
    /// </summary>
    /// <returns>Read-only list of registered rules</returns>
    public IReadOnlyList<IValidationRule> GetRules()
    {
        return _rules.AsReadOnly();
    }

    private void RegisterDefaultRules()
    {
        _rules.Add(new EmptyGraphRule());
        _rules.Add(new DynamicMemoryRule());
        _rules.Add(new ControlFlowRule());
        _rules.Add(new SynchronizationRule());
        _rules.Add(new IOOperationRule());
    }

    private int GetOperationCount(ICUDAGraph graph)
    {
        // Try to use the graph's Validate method to get operation count
        try
        {
            var validation = graph.Validate();
            return validation.OperationCount;
        }
        catch
        {
            return 0;
        }
    }

    private string GetErrorDescription(CudaError result)
    {
        return result switch
        {
            CudaError.InvalidValue => "Invalid parameter",
            CudaError.InvalidConfiguration => "Invalid configuration",
            CudaError.InvalidDevice => "Invalid device",
            CudaError.NotInitialized => "Not initialized",
            _ => $"CUDA error: {result}"
        };
    }
}
