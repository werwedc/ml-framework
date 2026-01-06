using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Abstract base class for user-defined custom autograd functions.
/// Enables manual implementation of forward and backward passes for specialized operations.
/// </summary>
public abstract class AutogradFunction
{
    private readonly List<Tensor> _savedTensors;
    private readonly List<object> _savedScalars;
    private OperationContext? _context;
    private bool _disposed;

    /// <summary>
    /// Gets the list of saved tensors for use during backward pass.
    /// </summary>
    protected List<Tensor> SavedTensors => _savedTensors;

    /// <summary>
    /// Gets the list of saved scalar values for use during backward pass.
    /// </summary>
    protected List<object> SavedScalars => _savedScalars;

    /// <summary>
    /// Gets the operation context for this function.
    /// </summary>
    protected OperationContext? Context => _context;

    /// <summary>
    /// Initializes a new instance of the AutogradFunction class.
    /// </summary>
    protected AutogradFunction()
    {
        _savedTensors = new List<Tensor>();
        _savedScalars = new List<object>();
    }

    /// <summary>
    /// Computes the forward pass of the operation.
    /// </summary>
    /// <param name="inputs">The input tensors to the operation.</param>
    /// <returns>The output tensor(s) of the operation.</returns>
    public abstract Tensor Forward(params Tensor[] inputs);

    /// <summary>
    /// Computes the backward pass of the operation.
    /// </summary>
    /// <param name="gradOutput">The gradient from the downstream operation.</param>
    /// <returns>The gradients for each input tensor.</returns>
    public abstract Tensor[] Backward(Tensor gradOutput);

    /// <summary>
    /// Saves tensors for use during the backward pass.
    /// </summary>
    /// <param name="tensors">The tensors to save.</param>
    protected void SaveForBackward(params Tensor[] tensors)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AutogradFunction));

        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        foreach (var tensor in tensors)
        {
            if (tensor == null)
                throw new ArgumentException("Cannot save null tensor");

            _savedTensors.Add(tensor);
        }
    }

    /// <summary>
    /// Saves a scalar value for use during the backward pass.
    /// </summary>
    /// <param name="scalar">The scalar value to save.</param>
    protected void SaveScalarForBackward(object scalar)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AutogradFunction));

        if (scalar == null)
            throw new ArgumentNullException(nameof(scalar));

        _savedScalars.Add(scalar);
    }

    /// <summary>
    /// Retrieves a saved tensor by index.
    /// </summary>
    /// <param name="index">The index of the saved tensor.</param>
    /// <returns>The saved tensor.</returns>
    protected Tensor GetSavedTensor(int index)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AutogradFunction));

        if (index < 0 || index >= _savedTensors.Count)
            throw new ArgumentOutOfRangeException(nameof(index),
                $"Index {index} is out of range. Saved tensors count: {_savedTensors.Count}");

        return _savedTensors[index];
    }

    /// <summary>
    /// Retrieves a saved scalar value by index and casts it to the specified type.
    /// </summary>
    /// <typeparam name="T">The type to cast the scalar to.</typeparam>
    /// <param name="index">The index of the saved scalar.</param>
    /// <returns>The saved scalar value cast to type T.</returns>
    protected T GetSavedScalar<T>(int index)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AutogradFunction));

        if (index < 0 || index >= _savedScalars.Count)
            throw new ArgumentOutOfRangeException(nameof(index),
                $"Index {index} is out of range. Saved scalars count: {_savedScalars.Count}");

        if (_savedScalars[index] is T typedValue)
            return typedValue;

        throw new InvalidCastException(
            $"Saved scalar at index {index} is of type {_savedScalars[index]?.GetType().Name}, not {typeof(T).Name}");
    }

    /// <summary>
    /// Creates an operation context for this function.
    /// </summary>
    /// <param name="operationName">The name of the operation.</param>
    /// <returns>The created operation context.</returns>
    public OperationContext CreateContext(string operationName)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AutogradFunction));

        if (string.IsNullOrEmpty(operationName))
            throw new ArgumentException("Operation name cannot be null or empty", nameof(operationName));

        _context = new OperationContext(operationName, ComputeGradients);
        return _context;
    }

    /// <summary>
    /// Applies this autograd function to the given inputs.
    /// </summary>
    /// <param name="inputs">The input tensors.</param>
    /// <returns>The output tensor.</returns>
    public Tensor Apply(params Tensor[] inputs)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AutogradFunction));

        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        // Clear previous state
        _savedTensors.Clear();
        _savedScalars.Clear();

        // Create context
        var operationName = GetType().Name.Replace("Function", "");
        _context = CreateContext(operationName);

        // Save inputs for backward pass if they require gradients
        foreach (var input in inputs)
        {
            if (input?.RequiresGrad == true)
            {
                // Input will be saved by user in Forward method
            }
        }

        // Compute forward pass
        var output = Forward(inputs);

        // Register with computational graph
        if (output.RequiresGrad)
        {
            var graph = GraphBuilder.GetCurrent();
            if (graph != null && graph.IsEnabled)
            {
                // Create graph nodes for children
                var childNodes = new List<GraphNode>();
                foreach (var input in inputs)
                {
                    if (input?.RequiresGrad == true && input.Parents != null)
                    {
                        // This is a simplification - in practice, you'd need to get the actual graph node
                        // For now, we'll create a leaf node
                        childNodes.Add(new GraphNode(input, _context));
                    }
                }

                var node = new GraphNode(output, _context, childNodes.ToArray());
                node.Register();
            }
        }

        return output;
    }

    /// <summary>
    /// Computes gradients for the saved tensors.
    /// </summary>
    /// <param name="gradOutput">The gradient from downstream.</param>
    /// <returns>The gradients for each input.</returns>
    private Tensor[] ComputeGradients(Tensor gradOutput)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(AutogradFunction));

        // Call the user's backward implementation
        var gradients = Backward(gradOutput);

        // Clear saved tensors after backward pass to free memory
        ClearSavedTensors();

        return gradients;
    }

    /// <summary>
    /// Clears all saved tensors and scalars.
    /// </summary>
    private void ClearSavedTensors()
    {
        _savedTensors.Clear();
        _savedScalars.Clear();
    }

    /// <summary>
    /// Disposes of the function's resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the function's resources.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                ClearSavedTensors();
                _context = null;
            }
            _disposed = true;
        }
    }

    /// <summary>
    /// Finalizer for AutogradFunction.
    /// </summary>
    ~AutogradFunction()
    {
        Dispose(false);
    }
}
