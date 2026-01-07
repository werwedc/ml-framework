using System.Collections.ObjectModel;

namespace MLFramework.Fusion
{
    /// <summary>
    /// Analyzes computational graph structure and properties
    /// </summary>
    public class GraphAnalyzer
    {
        /// <summary>
        /// Builds operation dependency graph from computational graph
        /// </summary>
        public DependencyGraph BuildDependencyGraph(ComputationalGraph graph)
        {
            var nodes = new Dictionary<string, DependencyNode>();

            // Create nodes for all operations
            foreach (var op in graph.Operations)
            {
                nodes[op.Id] = new DependencyNode
                {
                    Operation = op,
                    Dependencies = new List<string>(),
                    Dependents = new List<string>()
                };
            }

            // Connect nodes based on dependencies
            foreach (var op in graph.Operations)
            {
                var node = nodes[op.Id];
                foreach (var depId in op.GetDependencies())
                {
                    if (nodes.ContainsKey(depId))
                    {
                        node.Dependencies.Add(depId);
                        nodes[depId].Dependents.Add(op.Id);
                    }
                }
            }

            return new DependencyGraph
            {
                Nodes = new ReadOnlyDictionary<string, DependencyNode>(nodes)
            };
        }

        /// <summary>
        /// Identifies sequential operation chains without branching
        /// </summary>
        public List<OperationChain> FindLinearChains(DependencyGraph graph)
        {
            var chains = new List<OperationChain>();
            var visited = new HashSet<string>();
            var operationsById = new Dictionary<string, Operation>();

            // Build operation lookup
            foreach (var node in graph.Nodes.Values)
            {
                operationsById[node.Operation.Id] = node.Operation;
            }

            // Find operations with no dependencies (sources)
            var sources = graph.Nodes.Values
                .Where(n => n.Dependencies.Count == 0)
                .ToList();

            // Find chains starting from each source
            foreach (var source in sources)
            {
                if (visited.Contains(source.Operation.Id))
                    continue;

                var chain = ExtractLinearChain(graph, source.Operation.Id, operationsById, visited);
                if (chain != null && chain.Operations.Count > 1)
                {
                    chains.Add(chain);
                }
            }

            // Also check for chains that might not start from sources
            foreach (var node in graph.Nodes.Values)
            {
                if (visited.Contains(node.Operation.Id))
                    continue;

                // Skip nodes with multiple dependents (branching point)
                if (node.Dependents.Count > 1)
                    continue;

                var chain = ExtractLinearChain(graph, node.Operation.Id, operationsById, visited);
                if (chain != null && chain.Operations.Count > 1)
                {
                    chains.Add(chain);
                }
            }

            return chains;
        }

        private static OperationChain? ExtractLinearChain(
            DependencyGraph graph,
            string startOpId,
            Dictionary<string, Operation> operationsById,
            HashSet<string> visited)
        {
            var chainOps = new List<Operation>();
            var currentOpId = startOpId;
            var hasBranching = false;

            while (currentOpId != null)
            {
                if (visited.Contains(currentOpId))
                    break;

                var node = graph.Nodes[currentOpId];
                var op = node.Operation;

                chainOps.Add(op);
                visited.Add(currentOpId);

                // Check for branching
                if (node.Dependents.Count > 1)
                {
                    hasBranching = true;
                    break;
                }

                // Move to next operation
                currentOpId = node.Dependents.Count > 0 ? node.Dependents[0] : null;
            }

            if (chainOps.Count < 2)
                return null;

            return new OperationChain
            {
                Operations = chainOps,
                HasBranching = hasBranching
            };
        }

        /// <summary>
        /// Analyzes memory access patterns for operations
        /// </summary>
        public MemoryAccessPattern AnalyzeAccessPattern(Operation op)
        {
            return op.Type switch
            {
                "Add" or "Sub" or "Mul" or "Div" or "ReLU" or "Sigmoid" or
                "Tanh" or "LeakyReLU" or "Exp" or "Log" or "Sqrt" or "Abs"
                    => MemoryAccessPattern.ElementWise,

                "Conv2D" or "ConvTranspose2D" or "MaxPool2D" or "AvgPool2D"
                    => MemoryAccessPattern.Spatial,

                "ReduceSum" or "ReduceMean" or "ReduceMax" or "ReduceMin"
                    => MemoryAccessPattern.Reduction,

                "Gather" or "IndexSelect"
                    => MemoryAccessPattern.Gather,

                "Scatter" or "ScatterAdd"
                    => MemoryAccessPattern.Scatter,

                _ => MemoryAccessPattern.Unknown
            };
        }
    }

    /// <summary>
    /// Dependency graph representation of computational graph
    /// </summary>
    public class DependencyGraph
    {
        public required IReadOnlyDictionary<string, DependencyNode> Nodes { get; init; }
    }

    /// <summary>
    /// Node in dependency graph
    /// </summary>
    public class DependencyNode
    {
        public required Operation Operation { get; init; }
        public required List<string> Dependencies { get; init; }
        public required List<string> Dependents { get; init; }

        /// <summary>
        /// Gets operation thread block configuration if available
        /// </summary>
        public Backends.ThreadBlockConfiguration? ThreadBlockConfig =>
            Operation.Attributes.TryGetValue("threadBlockConfig", out var config) && config is Backends.ThreadBlockConfiguration tbc
                ? tbc
                : null;
    }

    /// <summary>
    /// Linear chain of operations
    /// </summary>
    public record OperationChain
    {
        public required IReadOnlyList<Operation> Operations { get; init; }
        public required bool HasBranching { get; init; }
    }

    /// <summary>
    /// Memory access patterns for operations
    /// </summary>
    public enum MemoryAccessPattern
    {
        /// <summary>
        /// Element-wise operations (same index across tensors)
        /// </summary>
        ElementWise,

        /// <summary>
        /// Spatial operations (convolutions, pooling)
        /// </summary>
        Spatial,

        /// <summary>
        /// Reduction operations (sum, mean, etc.)
        /// </summary>
        Reduction,

        /// <summary>
        /// Gather operations (index-based access)
        /// </summary>
        Gather,

        /// <summary>
        /// Scatter operations (index-based write)
        /// </summary>
        Scatter,

        /// <summary>
        /// Unknown or mixed pattern
        /// </summary>
        Unknown
    }
}
