using MLFramework.Core;
using MLFramework.Fusion.Backends;

namespace MLFramework.Fusion;

/// <summary>
/// Transforms computational graphs by applying fusion transformations
/// </summary>
public class GraphTransformer
{
    private readonly IFusionRegistry _registry;
    private readonly OperationCompatibilityChecker _compatibilityChecker;

    public GraphTransformer(IFusionRegistry registry)
    {
        _registry = registry;
        _compatibilityChecker = new OperationCompatibilityChecker();
    }

    /// <summary>
    /// Creates a fused operation from a list of operations
    /// </summary>
    public FusedOperation CreateFusedOperation(
        IReadOnlyList<Operation> operations,
        FusionPatternDefinition pattern)
    {
        var ir = BuildFusionIR(operations, pattern);
        var kernelSpec = EstimateKernelSpec(ir, pattern.Strategy);

        return FusedOperation.Create(
            id: Guid.NewGuid().ToString(),
            operations: operations,
            pattern: pattern,
            ir: ir,
            kernelSpec: kernelSpec);
    }

    /// <summary>
    /// Transforms a graph by applying fusion transformations
    /// </summary>
    public FusionResult TransformGraph(ComputationalGraph graph, FusionOptions options)
    {
        var fusedOps = new List<FusedOperation>();
        var rejectedFusions = new List<FusionRejected>();
        var processedOps = new HashSet<Operation>();

        if (!options.EnableFusion)
        {
            return new FusionResult
            {
                FusedGraph = graph,
                FusedOperations = fusedOps,
                OriginalOpCount = graph.Operations.Count,
                FusedOpCount = graph.Operations.Count,
                RejectedFusions = rejectedFusions
            };
        }

        // Find applicable fusion patterns
        var matches = _registry.FindMatches(graph.Operations);

        foreach (var match in matches)
        {
            if (match.MatchedOperations.Any(op => processedOps.Contains(op)))
                continue;

            // Check if chain exceeds max fusion ops
            if (match.MatchedOperations.Count > options.MaxFusionOps)
            {
                rejectedFusions.Add(new FusionRejected
                {
                    Operations = match.MatchedOperations.ToList(),
                    RejectionReason = $"Chain length ({match.MatchedOperations.Count}) exceeds MaxFusionOps ({options.MaxFusionOps})"
                });
                continue;
            }

            // Check compatibility score against minimum
            if (match.MatchScore < options.MinBenefitScore)
            {
                rejectedFusions.Add(new FusionRejected
                {
                    Operations = match.MatchedOperations.ToList(),
                    RejectionReason = $"Match score ({match.MatchScore}) below minimum ({options.MinBenefitScore})"
                });
                continue;
            }

            // Check pattern-specific constraints
            if (options.EnableConvActivationFusion && match.Pattern.Name.Contains("ConvActivation"))
                ; // Accept
            else if (options.EnableBatchNormFolding && match.Pattern.Name.Contains("BatchNorm"))
                ; // Accept
            else if (!_compatibilityChecker.AreSequenceCompatible(match.MatchedOperations))
            {
                rejectedFusions.Add(new FusionRejected
                {
                    Operations = match.MatchedOperations.ToList(),
                    RejectionReason = "Operations are not compatible"
                });
                continue;
            }

            // Create fused operation
            var fusedOp = CreateFusedOperation(match.MatchedOperations, match.Pattern);

            // Mark operations as processed
            foreach (var op in match.MatchedOperations)
                processedOps.Add(op);

            fusedOps.Add(fusedOp);
        }

        return new FusionResult
        {
            FusedGraph = graph,
            FusedOperations = fusedOps,
            OriginalOpCount = graph.Operations.Count,
            FusedOpCount = graph.Operations.Count - fusedOps.Sum(f => f.ConstituentOperations.Count - 1),
            RejectedFusions = rejectedFusions
        };
    }

    /// <summary>
    /// Builds fusion IR from operations and pattern
    /// </summary>
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
        if (variables.Count > 0)
        {
            variables[^1] = variables[^1] with { Location = MemoryLocation.Output };
        }

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

    /// <summary>
    /// Estimates kernel specification from IR and strategy
    /// </summary>
    private KernelSpecification EstimateKernelSpec(FusionIR ir, FusionStrategy strategy)
    {
        var inputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Input).ToList();
        var outputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Output).ToList();

        return new KernelSpecification
        {
            KernelName = $"fused_{ir.Id}",
            Strategy = strategy,
            InputTensors = inputTensors,
            OutputTensors = outputTensors,
            TemporaryMemoryBytes = ir.MemoryLayout.SharedMemoryBytes,
            RegisterBytes = ir.MemoryLayout.RegisterBytes,
            ThreadBlockConfig = new Backends.ThreadBlockConfiguration
            {
                X = ir.ComputeRequirements.ThreadsPerBlock,
                Y = 1,
                Z = 1
            },
            CompilationFlags = new List<string> { "-O3", "--use_fast_math" },
            Parameters = Array.Empty<KernelLaunchParameter>()
        };
    }

    /// <summary>
    /// Estimates register usage for an operation
    /// </summary>
    private int EstimateRegisterUsage(Operation op)
    {
        return op.Type switch
        {
            "Add" or "Sub" or "Mul" or "Div" => 4,
            "ReLU" => 2,
            "Sigmoid" or "Tanh" => 8,
            "Conv2D" => 32,
            "Linear" => 16,
            _ => 8
        };
    }

    /// <summary>
    /// Estimates number of thread blocks needed
    /// </summary>
    private int EstimateThreadBlocks(IReadOnlyList<Operation> operations)
    {
        var outputShape = operations[^1].OutputShape;
        var totalElements = outputShape.Dimensions.Aggregate(1, (a, b) => a * b);
        var threadsPerBlock = EstimateThreadsPerBlock(operations);
        return (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    }

    /// <summary>
    /// Estimates threads per block
    /// </summary>
    private int EstimateThreadsPerBlock(IReadOnlyList<Operation> operations)
    {
        // Default to 256 for most operations
        return operations[^1].Type switch
        {
            "Conv2D" => 128,
            "Linear" => 256,
            _ => 256
        };
    }
}
