# Spec: Graph Validation System

## Overview
Implement a comprehensive validation system for CUDA graphs that checks operations for graph compatibility and provides detailed feedback on potential issues. This helps users understand why certain operations cannot be captured in graphs.

## Requirements

### 1. CUDAGraphValidator Class
Implement the graph validation logic.

```csharp
public class CUDAGraphValidator
{
    private readonly List<IValidationRule> _rules;

    public CUDAGraphValidator()
    {
        _rules = new List<IValidationRule>();
        RegisterDefaultRules();
    }

    /// <summary>
    /// Validates a captured graph
    /// </summary>
    public CUDAGraphValidationResult Validate(ICUDAGraph graph)
    {
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
            var result = rule.Validate(graph);
            errors.AddRange(result.Errors);
            warnings.AddRange(result.Warnings);
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
    /// Validates a graph instance
    /// </summary>
    public CUDAGraphValidationResult ValidateGraphInstance(IntPtr graphHandle)
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        // Check if graph is null
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
        var result = CUDADriver.cuGraphInstantiate(
            out IntPtr graphExec,
            graphHandle,
            IntPtr.Zero,
            0);

        if (result != CUResult.Success)
        {
            errors.Add($"Graph instantiation failed: {GetErrorDescription(result)}");
        }

        // Get operation count
        var nodeResult = CUDADriver.cuGraphGetNodes(graphHandle, out _, out ulong nodeCount);
        if (nodeResult != CUResult.Success)
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
    public void RegisterRule(IValidationRule rule)
    {
        _rules.Add(rule);
    }

    /// <summary>
    /// Clears all validation rules
    /// </summary>
    public void ClearRules()
    {
        _rules.Clear();
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
        // Get operation count from graph
        if (graph is CUDAGraph cudaGraph)
        {
            var validation = cudaGraph.Validate();
            return validation.OperationCount;
        }
        return 0;
    }

    private string GetErrorDescription(CUResult result)
    {
        return result switch
        {
            CUResult.ErrorInvalidValue => "Invalid parameter",
            CUResult.ErrorNotSupported => "Operation not supported",
            CUResult.ErrorGraphExecUpdateFailure => "Graph exec update failed",
            _ => $"CUDA error: {result}"
        };
    }
}
```

### 2. IValidationRule Interface
Define the validation rule interface.

```csharp
public interface IValidationRule
{
    /// <summary>
    /// Validates the graph and returns any errors or warnings
    /// </summary>
    ValidationResult Validate(ICUDAGraph graph);
}

public class ValidationResult
{
    public List<string> Errors { get; }
    public List<string> Warnings { get; }

    public ValidationResult()
    {
        Errors = new List<string>();
        Warnings = new List<string>();
    }
}
```

### 3. Built-in Validation Rules
Implement standard validation rules.

```csharp
public class EmptyGraphRule : IValidationRule
{
    public ValidationResult Validate(ICUDAGraph graph)
    {
        var result = new ValidationResult();

        if (graph is CUDAGraph cudaGraph)
        {
            var validation = cudaGraph.Validate();
            if (validation.OperationCount == 0)
            {
                result.Errors.Add("Graph contains no operations to execute");
            }
        }

        return result;
    }
}

public class DynamicMemoryRule : IValidationRule
{
    public ValidationResult Validate(ICUDAGraph graph)
    {
        var result = new ValidationResult();

        // Check for dynamic memory allocations
        // This would require tracking memory operations during capture
        // For now, we'll add a warning
        result.Warnings.Add(
            "Ensure all memory is pre-allocated using the graph memory pool");

        return result;
    }
}

public class ControlFlowRule : IValidationRule
{
    public ValidationResult Validate(ICUDAGraph graph)
    {
        var result = new ValidationResult();

        // Check for data-dependent control flow
        // This would require analyzing captured operations
        result.Warnings.Add(
            "Avoid data-dependent control flow within graph capture");

        return result;
    }
}

public class SynchronizationRule : IValidationRule
{
    public ValidationResult Validate(ICUDAGraph graph)
    {
        var result = new ValidationResult();

        // Check for CPU-GPU synchronization points
        result.Warnings.Add(
            "Avoid CPU-GPU synchronization within graph capture");

        return result;
    }
}

public class IOOperationRule : IValidationRule
{
    public ValidationResult Validate(ICUDAGraph graph)
    {
        var result = new ValidationResult();

        // Check for I/O operations
        result.Warnings.Add(
            "Avoid I/O operations within graph capture");

        return result;
    }
}
```

### 4. CUDAGraphValidationContext
Provide context for validation.

```csharp
public class CUDAGraphValidationContext
{
    private readonly List<string> _capturedOperations;
    private readonly HashSet<IntPtr> _allocatedMemory;

    public CUDAGraphValidationContext()
    {
        _capturedOperations = new List<string>();
        _allocatedMemory = new HashSet<IntPtr>();
    }

    public void RecordOperation(string operationName)
    {
        _capturedOperations.Add(operationName);
    }

    public void RecordMemoryAllocation(IntPtr ptr)
    {
        _allocatedMemory.Add(ptr);
    }

    public IReadOnlyList<string> CapturedOperations => _capturedOperations;
    public IReadOnlySet<IntPtr> AllocatedMemory => _allocatedMemory;
}
```

## Implementation Details

### File Structure
- **File**: `src/CUDA/Graphs/Validation/CUDAGraphValidator.cs`
- **File**: `src/CUDA/Graphs/Validation/IValidationRule.cs`
- **File**: `src/CUDA/Graphs/Validation/Rules/EmptyGraphRule.cs`
- **File**: `src/CUDA/Graphs/Validation/Rules/DynamicMemoryRule.cs`
- **File**: `src/CUDA/Graphs/Validation/Rules/ControlFlowRule.cs`
- **File**: `src/CUDA/Graphs/Validation/Rules/SynchronizationRule.cs`
- **File**: `src/CUDA/Graphs/Validation/Rules/IOOperationRule.cs`
- **File**: `src/CUDA/Graphs/Validation/CUDAGraphValidationContext.cs`

### Dependencies
- ICUDAGraph interface (from spec_cuda_graph_core_interfaces)
- CUDAGraphValidationResult class (from spec_cuda_graph_core_interfaces)
- CUDAGraph class (from spec_cuda_graph_execution_engine)
- CUDADriver bindings (extend)
- System.Collections.Generic for List, HashSet

### Validation Strategy
1. **Structure Validation**: Check graph structure (empty, cycles, etc.)
2. **Operation Validation**: Check for unsupported operations
3. **Memory Validation**: Check for dynamic memory allocations
4. **Control Flow Validation**: Check for data-dependent branching
5. **Synchronization Validation**: Check for CPU-GPU sync points

### Error Messages
- Clear and actionable
- Include suggestions for fixes
- Differentiate between errors and warnings
- Provide context about where the issue occurred

## Success Criteria
- Validator can identify empty graphs
- Validator can detect unsupported operations
- Custom rules can be registered
- Validation provides clear error messages
- Context tracking works correctly
- Rules are applied in order

## Testing Requirements

### Unit Tests
- Test empty graph detection
- Test rule registration and execution
- Test custom rules
- Test error message generation
- Test warning generation
- Test context tracking

### Integration Tests
- Test validation with real graphs (requires GPU)
- Test validation with invalid operations
- Test validation across different graph types
