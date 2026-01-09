using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Represents a custom function node in the computational graph.
/// Stores the function instance, context, input/output tensors, and provides backward pass execution.
/// </summary>
public class CustomFunctionNode : IDisposable
{
    private bool _disposed;

    /// <summary>
    /// Gets or sets the custom function instance.
    /// </summary>
    public CustomFunction Function { get; set; } = null!;

    /// <summary>
    /// Gets or sets the function context saved during forward pass.
    /// </summary>
    public FunctionContext Context { get; set; } = null!;

    /// <summary>
    /// Gets or sets the input tensors to the function.
    /// </summary>
    public Tensor[] Inputs { get; set; } = null!;

    /// <summary>
    /// Gets or sets the output tensors from the function.
    /// </summary>
    public Tensor[] Outputs { get; set; } = null!;

    /// <summary>
    /// Gets the unique identifier for this node.
    /// </summary>
    public Guid Id { get; } = Guid.NewGuid();

    /// <summary>
    /// Gets whether this node has been disposed.
    /// </summary>
    public bool IsDisposed => _disposed;

    /// <summary>
    /// Executes the backward pass for this custom function.
    /// </summary>
    /// <param name="gradOutputs">Gradients with respect to the outputs.</param>
    /// <exception cref="InvalidOperationException">Thrown when node is disposed or context is null.</exception>
    public void Backward(Tensor[] gradOutputs)
    {
        if (_disposed)
            throw new InvalidOperationException("Cannot execute backward pass on disposed node");

        if (Context == null || Context.IsDisposed)
            throw new InvalidOperationException("Context is null or disposed");

        // Call the custom function's backward pass
        var gradInputs = Function.Backward(gradOutputs, Context);

        if (gradInputs == null)
            throw new InvalidOperationException("Backward pass returned null gradients");

        // Accumulate gradients on input tensors
        for (int i = 0; i < gradInputs.Length && i < Inputs.Length; i++)
        {
            if (Inputs[i].RequiresGrad && gradInputs[i] != null)
            {
                // Ensure gradient tensor exists before accumulating
                if (Inputs[i].Gradient == null)
                {
                    Inputs[i].Gradient = Tensor.Zeros(Inputs[i].Shape);
                }
                Inputs[i].AccumulateGrad(gradInputs[i]);
            }
        }
    }

    /// <summary>
    /// Disposes the node and its context, releasing resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes the node and its resources.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                Context?.Dispose();
            }

            Function = null!;
            Inputs = null!;
            Outputs = null!;
            _disposed = true;
        }
    }

    /// <summary>
    /// Finalizer for CustomFunctionNode.
    /// </summary>
    ~CustomFunctionNode()
    {
        Dispose(false);
    }
}
