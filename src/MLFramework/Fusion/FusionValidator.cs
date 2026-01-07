using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Validates correctness of fusion transformations
/// </summary>
public class FusionValidator
{
    /// <summary>
    /// Validates a fusion operation
    /// </summary>
    public FusionValidationResult ValidateFusion(
        FusedOperation fusedOp,
        ComputationalGraph originalGraph)
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        // Validate shape preservation
        if (!ValidateShapePreservation(fusedOp))
            errors.Add("Shape preservation validation failed");

        // Validate data flow
        if (!ValidateDataFlow(fusedOp))
            errors.Add("Data flow validation failed");

        // Validate memory safety
        ValidateMemorySafety(fusedOp, warnings);

        // Validate numerical stability
        ValidateNumericalStability(fusedOp, warnings);

        return new FusionValidationResult
        {
            IsValid = errors.Count == 0,
            Errors = errors,
            Warnings = warnings
        };
    }

    /// <summary>
    /// Validates that fusion preserves input and output shapes
    /// </summary>
    private bool ValidateShapePreservation(FusedOperation fusedOp)
    {
        var inputShape = fusedOp.ConstituentOperations[0].InputShape;
        var outputShape = fusedOp.ConstituentOperations[^1].OutputShape;

        return fusedOp.InputShape.Equals(inputShape) &&
               fusedOp.OutputShape.Equals(outputShape);
    }

    /// <summary>
    /// Validates data flow correctness in fusion IR
    /// </summary>
    private bool ValidateDataFlow(FusedOperation fusedOp)
    {
        var ir = fusedOp.IntermediateRepresentation;
        var graph = ir.BuildDataflowGraph();

        // Check for cycles
        if (graph.HasCycles())
            return false;

        // Check all variables are defined before use
        var definedVars = new HashSet<string>();
        definedVars.Add("input"); // Input is predefined

        foreach (var node in ir.Nodes)
        {
            // Check all input variables are defined
            foreach (var inputVar in node.InputVars)
            {
                if (!definedVars.Contains(inputVar) && inputVar != "input")
                {
                    return false;
                }
            }

            // Mark output variable as defined
            definedVars.Add(node.OutputVar);
        }

        // Check final output is defined
        if (ir.Nodes.Count > 0 && !definedVars.Contains(ir.Nodes[^1].OutputVar))
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Validates memory safety of fusion
    /// </summary>
    private void ValidateMemorySafety(FusedOperation fusedOp, List<string> warnings)
    {
        var ir = fusedOp.IntermediateRepresentation;

        // Check for excessive shared memory usage
        if (ir.ComputeRequirements.RequiresSharedMemory &&
            ir.MemoryLayout.SharedMemoryBytes > 48 * 1024)
        {
            warnings.Add($"Shared memory usage ({ir.MemoryLayout.SharedMemoryBytes} bytes) " +
                       "exceeds typical limits (48KB)");
        }

        // Check for excessive register usage
        const int maxRegistersPerThread = 255; // Typical limit
        var registersPerThread = (double)ir.MemoryLayout.RegisterBytes /
                               (4 * ir.ComputeRequirements.ThreadsPerBlock);

        if (registersPerThread > maxRegistersPerThread)
        {
            warnings.Add($"Register usage ({registersPerThread:F1} registers/thread) " +
                       "may exceed hardware limits");
        }
    }

    /// <summary>
    /// Validates numerical stability of fused operations
    /// </summary>
    private void ValidateNumericalStability(FusedOperation fusedOp, List<string> warnings)
    {
        foreach (var op in fusedOp.ConstituentOperations)
        {
            // Check for division by zero risks
            if (op.Type == "Div")
            {
                warnings.Add("Division operation may cause numerical instability " +
                           "if divisor is zero or very small");
            }

            // Check for log of non-positive numbers
            if (op.Type == "Log")
            {
                warnings.Add("Logarithm operation may cause numerical instability " +
                           "if input is non-positive");
            }

            // Check for operations that may overflow/underflow
            if (op.Type == "Exp" || op.Type == "Sigmoid")
            {
                if (op.DataType == DataType.Float16)
                {
                    warnings.Add($"Operation {op.Type} with Float16 may cause " +
                               "numerical overflow/underflow");
                }
            }
        }
    }

    /// <summary>
    /// Validates that fusion pattern is applicable
    /// </summary>
    public bool ValidatePatternApplicability(
        IReadOnlyList<Operation> operations,
        FusionPatternDefinition pattern)
    {
        // Check if number of operations matches pattern
        if (operations.Count > pattern.OpTypeSequence.Count)
        {
            return false;
        }

        // Check if all operations in pattern are present
        var opTypes = operations.Select(o => o.Type).ToHashSet();
        foreach (var requiredOp in pattern.OpTypeSequence)
        {
            if (!opTypes.Contains(requiredOp))
            {
                return false;
            }
        }

        return true;
    }
}
