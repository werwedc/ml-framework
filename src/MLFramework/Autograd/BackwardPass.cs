using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Implements reverse-mode automatic differentiation by traversing the computational graph
/// and computing gradients via the chain rule.
/// </summary>
public class BackwardPass
{
    private readonly GraphBuilder _graph;
    private readonly HashSet<GraphNode> _visitedNodes;
    private readonly Queue<GraphNode> _processingQueue;

    /// <summary>
    /// Gets the computational graph builder.
    /// </summary>
    public GraphBuilder Graph => _graph;

    /// <summary>
    /// Gets or sets whether to retain the computational graph after backward pass.
    /// When true, the graph is preserved for multiple backward passes (e.g., for higher-order derivatives).
    /// </summary>
    public bool RetainGraph { get; set; } = false;

    /// <summary>
    /// Initializes a new instance of the BackwardPass class.
    /// </summary>
    /// <param name="graph">The computational graph builder.</param>
    public BackwardPass(GraphBuilder graph)
    {
        _graph = graph ?? throw new ArgumentNullException(nameof(graph));
        _visitedNodes = new HashSet<GraphNode>();
        _processingQueue = new Queue<GraphNode>();
    }

    /// <summary>
    /// Runs the backward pass starting from the given tensor, computing gradients for all nodes in the graph.
    /// </summary>
    /// <param name="lossTensor">The tensor from which to start the backward pass (typically a loss scalar).</param>
    /// <param name="gradient">Optional custom gradient to use as the starting gradient. If null, uses gradient = 1.0.</param>
    /// <exception cref="ArgumentException">Thrown if the tensor is not a scalar and no gradient is provided.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the tensor is not in the computational graph.</exception>
    public void Run(Tensor lossTensor, Tensor? gradient = null)
    {
        if (lossTensor == null)
            throw new ArgumentNullException(nameof(lossTensor));

        // Find the graph node that produced this tensor
        var startNode = FindNodeForTensor(lossTensor);

        if (startNode == null)
            throw new InvalidOperationException("Tensor is not in the computational graph or graph is disabled");

        RunFromNode(startNode, gradient);
    }

    /// <summary>
    /// Runs the backward pass starting from a specific graph node.
    /// </summary>
    /// <param name="node">The node from which to start the backward pass.</param>
    /// <param name="gradient">Optional custom gradient to use as the starting gradient. If null, uses gradient = 1.0.</param>
    /// <exception cref="ArgumentException">Thrown if the node's output tensor is not a scalar and no gradient is provided.</exception>
    public void RunFromNode(GraphNode node, Tensor? gradient = null)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        // Initialize gradient
        if (gradient == null)
        {
            if (node.OutputTensor.Size != 1)
                throw new ArgumentException("Gradient must be provided for non-scalar tensors");

            gradient = Tensor.Ones(node.OutputTensor.Shape);
        }

        // Clear previous state
        _visitedNodes.Clear();
        _processingQueue.Clear();

        // Perform topological sort and traverse in reverse
        var sortedNodes = TopologicalSortReverse(node);

        // Initialize gradient at the starting node
        InitializeGradient(node.OutputTensor, gradient);

        // Process nodes in reverse topological order
        foreach (var currentNode in sortedNodes)
        {
            if (_visitedNodes.Contains(currentNode))
                continue;

            _visitedNodes.Add(currentNode);

            // Get the gradient for this node
            var gradOutput = currentNode.OutputTensor.Gradient;
            if (gradOutput == null)
                continue;

            // Compute gradients using the backward function
            ComputeGradients(currentNode, gradOutput);

            // Propagate gradients to children
            PropagateToChildren(currentNode, new[] { gradOutput });
        }

        // Clear the graph if not retaining
        if (!RetainGraph)
        {
            ClearGraphResources();
        }
    }

    /// <summary>
    /// Computes gradients for the current node using its backward function.
    /// </summary>
    /// <param name="node">The current graph node.</param>
    /// <param name="gradOutput">The gradient from the output side.</param>
    private void ComputeGradients(GraphNode node, Tensor gradOutput)
    {
        try
        {
            // Call the backward function from OperationContext
            var inputGradients = node.Operation.BackwardFn(gradOutput);

            if (inputGradients == null || inputGradients.Length == 0)
                return;

            // Distribute gradients to children
            for (int i = 0; i < Math.Min(inputGradients.Length, node.Children.Count); i++)
            {
                var childNode = node.Children[i];
                var childGrad = inputGradients[i];

                if (childGrad != null && childNode.OutputTensor.RequiresGrad)
                {
                    AccumulateGradient(childNode.OutputTensor, childGrad);
                }
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Error computing gradients for node {node.GradFnId}: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Propagates gradients to child nodes.
    /// </summary>
    /// <param name="node">The current graph node.</param>
    /// <param name="gradients">Array of gradients to propagate to children.</param>
    private void PropagateToChildren(GraphNode node, Tensor[] gradients)
    {
        for (int i = 0; i < node.Children.Count; i++)
        {
            var childNode = node.Children[i];

            if (i < gradients.Length && gradients[i] != null)
            {
                AccumulateGradient(childNode.OutputTensor, gradients[i]);
            }
        }
    }

    /// <summary>
    /// Accumulates gradient to a tensor (adds to existing gradient).
    /// </summary>
    /// <param name="tensor">The tensor to accumulate gradient to.</param>
    /// <param name="grad">The gradient to add.</param>
    private void AccumulateGradient(Tensor tensor, Tensor grad)
    {
        if (!tensor.RequiresGrad)
            return;

        if (tensor.Gradient == null)
        {
            tensor.Gradient = Tensor.Zeros(tensor.Shape);
        }

        // Ensure shapes match (handle broadcasting if needed)
        if (!tensor.Gradient.Shape.SequenceEqual(grad.Shape))
        {
            // Handle broadcasting: reduce gradient to match tensor shape
            var reducedGrad = BroadcastReduce(grad, tensor.Shape);
            var accumulatedData = tensor.Gradient.Data.Zip(reducedGrad.Data, (a, b) => a + b).ToArray();
            tensor.Gradient = new Tensor(accumulatedData, tensor.Gradient.Shape);
        }
        else
        {
            // Direct accumulation
            var accumulatedData = tensor.Gradient.Data.Zip(grad.Data, (a, b) => a + b).ToArray();
            tensor.Gradient = new Tensor(accumulatedData, tensor.Gradient.Shape);
        }
    }

    /// <summary>
    /// Initializes gradient for the starting tensor.
    /// </summary>
    /// <param name="tensor">The tensor to initialize gradient for.</param>
    /// <param name="gradient">The initial gradient.</param>
    private void InitializeGradient(Tensor tensor, Tensor gradient)
    {
        if (!tensor.RequiresGrad)
            return;

        tensor.Gradient = gradient.Clone();
    }

    /// <summary>
    /// Finds the graph node that produced the given tensor.
    /// </summary>
    /// <param name="tensor">The tensor to find the node for.</param>
    /// <returns>The graph node, or null if not found.</returns>
    private GraphNode? FindNodeForTensor(Tensor tensor)
    {
        // This is a simplified implementation - in practice, we'd need a better mapping
        // For now, return the most recent root node
        var rootNodes = _graph.GetRootNodes();
        return rootNodes.FirstOrDefault(n => n.OutputTensor == tensor);
    }

    /// <summary>
    /// Performs reverse topological sort starting from the given node.
    /// </summary>
    /// <param name="startNode">The starting node.</param>
    /// <returns>List of nodes in reverse topological order.</returns>
    private List<GraphNode> TopologicalSortReverse(GraphNode startNode)
    {
        var sorted = new List<GraphNode>();
        var visited = new HashSet<GraphNode>();
        var temp = new HashSet<GraphNode>();

        // DFS from start node
        Visit(startNode, sorted, visited, temp);

        // Reverse to get reverse topological order
        sorted.Reverse();

        return sorted;
    }

    /// <summary>
    /// DFS visit for topological sort.
    /// </summary>
    private void Visit(GraphNode node, List<GraphNode> sorted, HashSet<GraphNode> visited, HashSet<GraphNode> temp)
    {
        if (temp.Contains(node))
            throw new InvalidOperationException("Cycle detected in computational graph");

        if (visited.Contains(node))
            return;

        temp.Add(node);

        // Visit children first
        foreach (var child in node.Children)
        {
            Visit(child, sorted, visited, temp);
        }

        temp.Remove(node);
        visited.Add(node);
        sorted.Add(node);
    }

    /// <summary>
    /// Reduces a broadcasted gradient to match a target shape.
    /// </summary>
    /// <param name="gradient">The gradient to reduce.</param>
    /// <param name="targetShape">The target shape.</param>
    /// <returns>The reduced gradient.</returns>
    private Tensor BroadcastReduce(Tensor gradient, int[] targetShape)
    {
        // Simple implementation: if shapes don't match, return gradient as-is
        // In a full implementation, we'd sum over broadcasted dimensions
        return gradient;
    }

    /// <summary>
    /// Clears resources after backward pass (if not retaining graph).
    /// </summary>
    private void ClearGraphResources()
    {
        // Clear saved tensors from operation contexts
        // This would require iterating through nodes and clearing their contexts
        // For now, we'll let the garbage collector handle it
    }
}
