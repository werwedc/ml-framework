# Spec: Fusion Engine Core

## Overview
Implement the fusion engine that transforms computational graphs by merging compatible operations into fused nodes, generating intermediate representations for the fused kernels.

## Requirements

### 1. Fusion Engine Interface
Core interface for performing graph fusion.

```csharp
public interface IFusionEngine
{
    /// <summary>
    /// Applies fusion transformations to a computational graph
    /// </summary>
    FusionResult ApplyFusion(ComputationalGraph graph, FusionOptions options);

    /// <summary>
    /// Fuses a specific set of operations in the graph
    /// </summary>
    FusedOperation FuseOperations(IReadOnlyList<Operation> operations, FusionPattern pattern);

    /// <summary>
    /// Validates that a fusion transformation is correct
    /// </summary>
    FusionValidationResult ValidateFusion(FusedOperation fusedOp, ComputationalGraph originalGraph);
}

public record FusionResult
{
    public required ComputationalGraph FusedGraph { get; init; }
    public required IReadOnlyList<FusedOperation> FusedOperations { get; init; }
    public required int OriginalOpCount { get; init; }
    public required int FusedOpCount { get; init; }
    public required IReadOnlyList<FusionRejected> RejectedFusions { get; init; }
}

public record FusionRejected
{
    public required IReadOnlyList<Operation> Operations { get; init; }
    public required string RejectionReason { get; init; }
}

public record FusionValidationResult
{
    public required bool IsValid { get; init; }
    public required IReadOnlyList<string> Errors { get; init; }
    public required IReadOnlyList<string> Warnings { get; init; }
}

public record FusionOptions
{
    public bool EnableFusion { get; init; } = true;
    public int MaxFusionOps { get; init; } = 10;
    public int MinBenefitScore { get; init; } = 50;
    public bool EnableBatchNormFolding { get; init; } = true;
    public bool EnableConvActivationFusion { get; init; } = true;
    public FusionAggressiveness Aggressiveness { get; init; } = FusionAggressiveness.Medium;
}

public enum FusionAggressiveness
{
    Conservative,
    Medium,
    Aggressive
}
```

### 2. Fused Operation Representation
Representation of a fused operation in the IR.

```csharp
public class FusedOperation : Operation
{
    public required IReadOnlyList<Operation> ConstituentOperations { get; init; }
    public required FusionPattern Pattern { get; init; }
    public required FusionIR IntermediateRepresentation { get; init; }
    public required KernelSpecification KernelSpec { get; init; }

    public override string Name => $"Fused_{Pattern.Name}_{Id}";

    public FusedOperation(
        string id,
        IReadOnlyList<Operation> operations,
        FusionPattern pattern,
        FusionIR ir,
        KernelSpecification kernelSpec)
        : base(id, $"Fused_{pattern.Name}")
    {
        ConstituentOperations = operations;
        Pattern = pattern;
        IntermediateRepresentation = ir;
        KernelSpec = kernelSpec;
        InputShape = operations[0].InputShape;
        OutputShape = operations[^1].OutputShape;
    }
}
```

### 3. Fusion Intermediate Representation
IR for representing fused operations before kernel generation.

```csharp
public class FusionIR
{
    public required string Id { get; init; }
    public required IReadOnlyList<FusionOpNode> Nodes { get; init; }
    public required IReadOnlyList<FusionVariable> Variables { get; init; }
    public required MemoryLayout MemoryLayout { get; init; }
    public required ComputeRequirements ComputeRequirements { get; init; }

    /// <summary>
    /// Gets the dataflow graph of the fusion
    /// </summary>
    public FusionDataflowGraph BuildDataflowGraph();
}

public record FusionOpNode
{
    public required string Id { get; init; }
    public required string OriginalOpType { get; init; }
    public required IReadOnlyList<string> InputVars { get; init; }
    public required string OutputVar { get; init; }
    public required Dictionary<string, object> Attributes { get; init; }
}

public record FusionVariable
{
    public required string Name { get; init; }
    public required TensorShape Shape { get; init; }
    public required TensorDataType DataType { get; init; }
    public required MemoryLocation Location { get; init; }
}

public enum MemoryLocation
{
    Input,      // Input tensor
    Output,     // Output tensor
    Temporary,  // Temporary buffer in shared memory
    Register    // Stored in registers
}

public record MemoryLayout
{
    public required TensorLayout TensorLayout { get; init; }
    public required int SharedMemoryBytes { get; init; }
    public required int RegisterBytes { get; init; }
}

public record ComputeRequirements
{
    public required int ThreadBlocks { get; init; }
    public required int ThreadsPerBlock { get; init; }
    public required bool RequiresSharedMemory { get; init; }
    public required bool RequiresAtomicOps { get; init; }
}
```

### 4. Graph Transformer
Performs the actual graph transformation.

```csharp
public class GraphTransformer
{
    private readonly IFusionRegistry _registry;
    private readonly OperationCompatibilityChecker _compatibilityChecker;

    public GraphTransformer(IFusionRegistry registry)
    {
        _registry = registry;
        _compatibilityChecker = new OperationCompatibilityChecker();
    }

    public FusionResult TransformGraph(ComputationalGraph graph, FusionOptions options)
    {
        var fusedOps = new List<FusedOperation>();
        var rejectedFusions = new List<FusionRejected>();
        var processedOps = new HashSet<Operation>();
        var newGraph = graph.Clone();

        // Find linear chains in the graph
        var chains = new GraphAnalyzer().FindLinearChains(graph.DependencyGraph);

        foreach (var chain in chains)
        {
            if (chain.Operations.Any(op => processedOps.Contains(op)))
                continue;

            // Find applicable fusion patterns
            var matches = _registry.FindMatches(chain.Operations);

            if (matches.Count == 0)
                continue;

            // Select best match based on benefit score
            var bestMatch = matches.OrderByDescending(m => m.MatchScore).First();

            // Apply constraints
            if (!FusionSatisfiesConstraints(bestMatch, options))
            {
                rejectedFusions.Add(new FusionRejected
                {
                    Operations = chain.Operations.ToList(),
                    RejectionReason = "Does not satisfy fusion constraints"
                });
                continue;
            }

            // Create fused operation
            var fusedOp = CreateFusedOperation(chain.Operations, bestMatch.Pattern);

            // Replace chain with fused operation in graph
            newGraph.ReplaceChain(chain.Operations, fusedOp);

            fusedOps.Add(fusedOp);
            foreach (var op in chain.Operations)
                processedOps.Add(op);
        }

        return new FusionResult
        {
            FusedGraph = newGraph,
            FusedOperations = fusedOps,
            OriginalOpCount = graph.Operations.Count,
            FusedOpCount = newGraph.Operations.Count,
            RejectedFusions = rejectedFusions
        };
    }

    private FusedOperation CreateFusedOperation(
        IReadOnlyList<Operation> operations,
        FusionPatternDefinition pattern)
    {
        var ir = BuildFusionIR(operations, pattern);
        var kernelSpec = EstimateKernelSpec(ir, pattern.Strategy);

        return new FusedOperation(
            id: Guid.NewGuid().ToString(),
            operations: operations,
            pattern: pattern,
            ir: ir,
            kernelSpec: kernelSpec);
    }

    private FusionIR BuildFusionIR(
        IReadOnlyList<Operation> operations,
        FusionPatternDefinition pattern)
    {
        var nodes = new List<FusionOpNode>();
        var variables = new List<FusionVariable>();
        var input = operations[0].InputShape;
        int regBytes = 0;
        int sharedMemBytes = 0;

        // Create input variable
        var inputVar = new FusionVariable
        {
            Name = "input",
            Shape = input,
            DataType = operations[0].DataType,
            Location = MemoryLocation.Input
        };
        variables.Add(inputVar);

        string currentVar = "input";
        int nodeCounter = 0;

        foreach (var op in operations)
        {
            var outputVar = $"var_{nodeCounter}";
            var node = new FusionOpNode
            {
                Id = $"node_{nodeCounter}",
                OriginalOpType = op.Type,
                InputVars = new[] { currentVar },
                OutputVar = outputVar,
                Attributes = op.Attributes.ToDictionary()
            };
            nodes.Add(node);

            var outVar = new FusionVariable
            {
                Name = outputVar,
                Shape = op.OutputShape,
                DataType = op.DataType,
                Location = MemoryLocation.Temporary
            };
            variables.Add(outVar);

            regBytes += EstimateRegisterUsage(op);
            currentVar = outputVar;
            nodeCounter++;
        }

        // Mark final output
        variables[^1] = variables[^1] with { Location = MemoryLocation.Output };

        return new FusionIR
        {
            Id = $"ir_{Guid.NewGuid().ToString().Substring(0, 8)}",
            Nodes = nodes,
            Variables = variables,
            MemoryLayout = new MemoryLayout
            {
                TensorLayout = operations[0].Layout,
                SharedMemoryBytes = sharedMemBytes,
                RegisterBytes = regBytes
            },
            ComputeRequirements = new ComputeRequirements
            {
                ThreadBlocks = EstimateThreadBlocks(operations),
                ThreadsPerBlock = EstimateThreadsPerBlock(operations),
                RequiresSharedMemory = sharedMemBytes > 0,
                RequiresAtomicOps = false
            }
        };
    }

    private KernelSpecification EstimateKernelSpec(FusionIR ir, FusionStrategy strategy)
    {
        return new KernelSpecification
        {
            KernelName = $"fused_{ir.Id}",
            Strategy = strategy,
            InputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Input).ToList(),
            OutputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Output).ToList(),
            TemporaryMemoryBytes = ir.MemoryLayout.SharedMemoryBytes,
            RegisterBytes = ir.MemoryLayout.RegisterBytes
        };
    }
}
```

### 5. Fusion Validator
Validates correctness of fusion transformations.

```csharp
public class FusionValidator
{
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

    private bool ValidateShapePreservation(FusedOperation fusedOp)
    {
        var inputShape = fusedOp.ConstituentOperations[0].InputShape;
        var outputShape = fusedOp.ConstituentOperations[^1].OutputShape;

        return fusedOp.InputShape.Equals(inputShape) &&
               fusedOp.OutputShape.Equals(outputShape);
    }

    private bool ValidateDataFlow(FusedOperation fusedOp)
    {
        var ir = fusedOp.IntermediateRepresentation;
        var graph = ir.BuildDataflowGraph();

        // Check for cycles
        if (graph.HasCycles())
            return false;

        // Check all variables are defined
        var definedVars = new HashSet<string>();
        foreach (var node in ir.Nodes)
        {
            foreach (var inputVar in node.InputVars)
            {
                if (!definedVars.Contains(inputVar) && inputVar != "input")
                    return false;
            }
            definedVars.Add(node.OutputVar);
        }

        return true;
    }

    private void ValidateMemorySafety(FusedOperation fusedOp, List<string> warnings)
    {
        // Check for potential memory access conflicts
        if (fusedOp.IntermediateRepresentation.ComputeRequirements.RequiresSharedMemory &&
            fusedOp.IntermediateRepresentation.MemoryLayout.SharedMemoryBytes > 48 * 1024)
        {
            warnings.Add("Shared memory usage exceeds typical limits (48KB)");
        }
    }

    private void ValidateNumericalStability(FusedOperation fusedOp, List<string> warnings)
    {
        // Check for operations that might cause numerical issues
        foreach (var op in fusedOp.ConstituentOperations)
        {
            if (op.Type == "Div" || op.Type == "Log")
            {
                warnings.Add($"Operation {op.Type} may cause numerical instability");
            }
        }
    }
}
```

## Implementation Tasks

1. **Create fusion engine interfaces** (20 min)
   - IFusionEngine interface
   - FusionResult record
   - FusedOperation class
   - FusionOptions and enums

2. **Implement FusionIR data structures** (25 min)
   - FusionIR class
   - FusionOpNode record
   - FusionVariable record
   - MemoryLayout and ComputeRequirements

3. **Implement GraphTransformer core** (30 min)
   - TransformGraph method
   - Find and select best fusion matches
   - Replace chains with fused operations

4. **Implement IR building logic** (25 min)
   - BuildFusionIR method
   - Create variable definitions
   - Track data flow through operations
   - Estimate memory and compute requirements

5. **Implement FusionValidator** (25 min)
   - Validate shape preservation
   - Validate data flow
   - Validate memory safety
   - Validate numerical stability

6. **Implement resource estimation helpers** (20 min)
   - EstimateRegisterUsage
   - EstimateThreadBlocks
   - EstimateThreadsPerBlock
   - Fusion constraint checking

## Test Cases

```csharp
[Test]
public void TransformGraph_FusesElementWiseChain()
{
    var graph = BuildGraph(new[] { "Add", "Mul", "ReLU" });
    var engine = new FusionEngine(CreateRegistry());
    var options = new FusionOptions { EnableFusion = true };

    var result = engine.ApplyFusion(graph, options);

    Assert.IsTrue(result.FusedOpCount < result.OriginalOpCount);
    Assert.AreEqual(1, result.FusedOperations.Count);
}

[Test]
public void ValidateFusion_ValidChain_ReturnsTrue()
{
    var fusedOp = CreateFusedOperation(new[] { CreateAddOp(), CreateReluOp() });
    var validator = new FusionValidator();

    var result = validator.ValidateFusion(fusedOp, CreateOriginalGraph());

    Assert.IsTrue(result.IsValid);
    Assert.IsEmpty(result.Errors);
}

[Test]
public void ValidateFusion_ShapeMismatch_ReturnsFalse()
{
    var fusedOp = CreateFusedOpWithShapeMismatch();
    var validator = new FusionValidator();

    var result = validator.ValidateFusion(fusedOp, CreateOriginalGraph());

    Assert.IsFalse(result.IsValid);
    Assert.IsNotEmpty(result.Errors);
}

[Test]
public void TransformGraph_RespectsMaxFusionOps()
{
    var graph = BuildGraph(new[] { "Add", "Mul", "ReLU", "Sigmoid", "Tanh" });
    var options = new FusionOptions { MaxFusionOps = 3 };
    var engine = new FusionEngine(CreateRegistry());

    var result = engine.ApplyFusion(graph, options);

    // Should not fuse more than 3 operations
    var maxFused = result.FusedOperations.Max(op => op.ConstituentOperations.Count);
    Assert.LessOrEqual(maxFused, 3);
}
```

## Success Criteria
- Graph transformer correctly identifies and fuses operation chains
- Fusion IR accurately represents dataflow and resource requirements
- Validator catches invalid transformations
- Fusion options control fusion behavior correctly
- Original graph structure is preserved in unfused regions

## Dependencies
- FusionPatternRegistry
- GraphAnalyzer
- OperationCompatibilityChecker
- ComputationalGraph IR
