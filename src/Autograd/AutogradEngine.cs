using System.Collections.Concurrent;
using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Singleton autograd engine that manages the computational graph and custom function registration.
/// Handles backward pass traversal and gradient computation.
/// </summary>
public sealed class AutogradEngine
{
    private static readonly Lazy<AutogradEngine> _instance = new Lazy<AutogradEngine>(() => new AutogradEngine());
    private readonly ConcurrentDictionary<Guid, CustomFunctionNode> _nodes = new ConcurrentDictionary<Guid, CustomFunctionNode>();
    private readonly object _lock = new object();

    /// <summary>
    /// Gets the singleton instance of the autograd engine.
    /// </summary>
    public static AutogradEngine Instance => _instance.Value;

    /// <summary>
    /// Private constructor for singleton pattern.
    /// </summary>
    private AutogradEngine()
    {
    }

    /// <summary>
    /// Registers a custom function invocation with the autograd engine.
    /// Creates a graph node, attaches backward functions to outputs, and adds the node to the graph.
    /// </summary>
    /// <param name="outputs">Output tensors from the function.</param>
    /// <param name="function">The custom function instance.</param>
    /// <param name="context">Function context from the forward pass.</param>
    /// <param name="inputs">Input tensors to the function.</param>
    public void RegisterCustomFunction(
        Tensor[] outputs,
        CustomFunction function,
        FunctionContext context,
        Tensor[] inputs)
    {
        if (outputs == null)
            throw new ArgumentNullException(nameof(outputs));
        if (function == null)
            throw new ArgumentNullException(nameof(function));
        if (context == null)
            throw new ArgumentNullException(nameof(context));
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        var node = new CustomFunctionNode
        {
            Function = function,
            Context = context,
            Inputs = inputs,
            Outputs = outputs
        };

        // Attach backward function to outputs that require grad
        for (int i = 0; i < outputs.Length; i++)
        {
            if (outputs[i].RequiresGrad)
            {
                outputs[i].SetGradFn(CreateBackwardFunction(node, i));
            }
        }

        // Add node to graph
        AddNode(node);
    }

    /// <summary>
    /// Creates a backward function for a specific output tensor.
    /// </summary>
    /// <param name="node">The custom function node.</param>
    /// <param name="outputIndex">The index of the output tensor.</param>
    /// <returns>A function that computes gradients when called.</returns>
    private Func<Tensor[], Tensor[]> CreateBackwardFunction(CustomFunctionNode node, int outputIndex)
    {
        return (Tensor[] upstreamGrads) =>
        {
            // Call the custom function's backward pass
            var gradOutputs = new Tensor[node.Outputs.Length];
            gradOutputs[outputIndex] = upstreamGrads[0];

            var gradInputs = node.Function.Backward(gradOutputs, node.Context);

            return gradInputs;
        };
    }

    /// <summary>
    /// Adds a node to the computational graph.
    /// </summary>
    /// <param name="node">The node to add.</param>
    private void AddNode(CustomFunctionNode node)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        _nodes.TryAdd(node.Id, node);
    }

    /// <summary>
    /// Executes backward pass from the given output tensor.
    /// Traverses the computational graph and computes gradients for all leaf tensors.
    /// </summary>
    /// <param name="output">The output tensor to start backward pass from.</param>
    /// <param name="initialGrad">Optional initial gradient (defaults to ones like the output).</param>
    public void Backward(Tensor output, Tensor? initialGrad = null)
    {
        if (output == null)
            throw new ArgumentNullException(nameof(output));

        var queue = new Queue<(Tensor, CustomFunctionNode?, int)>();
        var visited = new HashSet<Guid>();
        var visitedNodes = new HashSet<Guid>();

        if (initialGrad != null)
        {
            output.AccumulateGrad(initialGrad);
        }
        else
        {
            if (output.Size != 1)
                throw new ArgumentException("Gradient must be provided for non-scalar tensors");
            output.AccumulateGrad(Tensor.Ones(output.Shape));
        }

        // Find the node that produced this output
        var node = FindNodeForOutput(output);
        if (node != null)
        {
            queue.Enqueue((output, node, 0));
            visited.Add(output.Id);
            visitedNodes.Add(node.Id);
        }

        while (queue.Count > 0)
        {
            var (tensor, currentNode, outputIndex) = queue.Dequeue();

            if (currentNode != null && !visitedNodes.Contains(currentNode.Id))
            {
                // Mark node as visited
                visitedNodes.Add(currentNode.Id);

                // Create gradient array for all outputs of this node
                var gradOutputs = new Tensor[currentNode.Outputs.Length];
                for (int i = 0; i < currentNode.Outputs.Length; i++)
                {
                    var outputTensor = currentNode.Outputs[i];

                    // Use gradient if set, otherwise use zeros
                    if (outputTensor.Gradient != null)
                    {
                        gradOutputs[i] = outputTensor.Gradient;
                    }
                    else
                    {
                        gradOutputs[i] = Tensor.Zeros(outputTensor.Shape);
                    }
                }

                // Call the node's backward method directly
                currentNode.Backward(gradOutputs);

                // Propagate to input tensors
                for (int i = 0; i < currentNode.Inputs.Length; i++)
                {
                    var inputTensor = currentNode.Inputs[i];

                    if (inputTensor.RequiresGrad)
                    {
                        // Gradient was accumulated in currentNode.Backward
                        if (!visited.Contains(inputTensor.Id))
                        {
                            visited.Add(inputTensor.Id);
                            var inputNode = FindNodeForOutput(inputTensor);
                            queue.Enqueue((inputTensor, inputNode, 0));
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Finds the node that produced the given output tensor.
    /// </summary>
    /// <param name="output">The output tensor to find the node for.</param>
    /// <returns>The custom function node, or null if not found.</returns>
    private CustomFunctionNode? FindNodeForOutput(Tensor output)
    {
        foreach (var kvp in _nodes)
        {
            if (kvp.Value.Outputs.Contains(output))
            {
                return kvp.Value;
            }
        }
        return null;
    }

    /// <summary>
    /// Clears all nodes from the computational graph and disposes them.
    /// Useful for freeing memory between training iterations.
    /// </summary>
    public void ClearGraph()
    {
        lock (_lock)
        {
            foreach (var node in _nodes.Values)
            {
                node.Dispose();
            }
            _nodes.Clear();
        }
    }

    /// <summary>
    /// Gets the number of nodes currently in the computational graph.
    /// </summary>
    public int NodeCount => _nodes.Count;
}
