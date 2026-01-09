using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Provides the mechanism to save and retrieve state (tensors and objects) between forward and backward passes in custom autograd functions.
/// </summary>
public class FunctionContext : IDisposable
{
    private List<Tensor> _savedTensors = new List<Tensor>();
    private List<object> _savedObjects = new List<object>();
    private bool _disposed = false;

    /// <summary>
    /// Gets the number of saved tensors.
    /// </summary>
    public int SavedTensorCount => _savedTensors.Count;

    /// <summary>
    /// Gets the number of saved objects.
    /// </summary>
    public int SavedObjectCount => _savedObjects.Count;

    /// <summary>
    /// Gets whether the context has been disposed.
    /// </summary>
    public bool IsDisposed => _disposed;

    /// <summary>
    /// Saves tensors for retrieval during the backward pass.
    /// </summary>
    /// <param name="tensors">The tensors to save.</param>
    public void SaveForBackward(params Tensor?[]? tensors)
    {
        if (_disposed)
        {
            throw new InvalidOperationException("Cannot save tensors from disposed context");
        }

        if (tensors == null)
        {
            // When null is passed to params array, interpret as empty array
            return;
        }

        foreach (var tensor in tensors)
        {
            _savedTensors.Add(tensor);
        }
    }

    /// <summary>
    /// Saves objects (scalars, integers, strings, etc.) for retrieval during the backward pass.
    /// </summary>
    /// <param name="objects">The objects to save.</param>
    public void SaveForBackward(params object?[]? objects)
    {
        if (_disposed)
        {
            throw new InvalidOperationException("Cannot save objects from disposed context");
        }

        if (objects == null)
        {
            // When null is passed to params array, interpret as empty array
            return;
        }

        foreach (var obj in objects)
        {
            _savedObjects.Add(obj);
        }
    }

    /// <summary>
    /// Retrieves a saved tensor at the specified index.
    /// </summary>
    /// <param name="index">The index of the tensor to retrieve.</param>
    /// <returns>The saved tensor.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the context is disposed.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if the index is out of range.</exception>
    public Tensor GetSavedTensor(int index)
    {
        if (_disposed)
        {
            throw new InvalidOperationException("Cannot retrieve tensor from disposed context");
        }

        if (index < 0 || index >= _savedTensors.Count)
        {
            throw new ArgumentOutOfRangeException(
                nameof(index),
                index,
                $"Tensor index {index} is out of range. Valid range is 0 to {_savedTensors.Count - 1}");
        }

        return _savedTensors[index];
    }

    /// <summary>
    /// Retrieves a saved object at the specified index.
    /// </summary>
    /// <param name="index">The index of the object to retrieve.</param>
    /// <returns>The saved object.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the context is disposed.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if the index is out of range.</exception>
    public object GetSavedObject(int index)
    {
        if (_disposed)
        {
            throw new InvalidOperationException("Cannot retrieve object from disposed context");
        }

        if (index < 0 || index >= _savedObjects.Count)
        {
            throw new ArgumentOutOfRangeException(
                nameof(index),
                index,
                $"Object index {index} is out of range. Valid range is 0 to {_savedObjects.Count - 1}");
        }

        return _savedObjects[index];
    }

    /// <summary>
    /// Clears all saved tensors and objects, releasing references.
    /// </summary>
    public void Clear()
    {
        _savedTensors.Clear();
        _savedObjects.Clear();
    }

    /// <summary>
    /// Disposes the context and clears all saved state.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            Clear();
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}
