using RitterFramework.Core.Tensor;
using System.Collections.Concurrent;

namespace MLFramework.Autograd;

/// <summary>
/// Stores operation metadata, saved intermediate tensors, and the backward function
/// needed for gradient computation during the backward pass.
/// </summary>
public class OperationContext
{
    private static int _nextId = 0;
    private readonly object _lock = new object();
    private readonly Dictionary<string, object> _savedTensors;
    private bool _disposed = false;

    /// <summary>
    /// Gets the name of this operation.
    /// </summary>
    public string OperationName { get; }

    /// <summary>
    /// Gets the dictionary of saved tensors/objects needed for backward pass.
    /// </summary>
    public IReadOnlyDictionary<string, object> SavedTensors => _savedTensors;

    /// <summary>
    /// Gets the backward function that computes gradients.
    /// </summary>
    public Func<Tensor, Tensor[]> BackwardFn { get; }

    /// <summary>
    /// Gets the unique identifier for this operation.
    /// </summary>
    public int OperationId { get; }

    /// <summary>
    /// Initializes a new instance of the OperationContext class.
    /// </summary>
    /// <param name="name">The name of the operation.</param>
    /// <param name="backwardFn">The backward function for gradient computation.</param>
    public OperationContext(string name, Func<Tensor, Tensor[]> backwardFn)
    {
        OperationName = name ?? throw new ArgumentNullException(nameof(name));
        BackwardFn = backwardFn ?? throw new ArgumentNullException(nameof(backwardFn));
        _savedTensors = new Dictionary<string, object>();
        OperationId = Interlocked.Increment(ref _nextId);
    }

    /// <summary>
    /// Saves a tensor or object for use during the backward pass.
    /// </summary>
    /// <param name="key">The key to identify the saved tensor/object.</param>
    /// <param name="tensor">The tensor or object to save.</param>
    public void SaveTensor(string key, object tensor)
    {
        if (string.IsNullOrEmpty(key))
            throw new ArgumentException("Key cannot be null or empty", nameof(key));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (_disposed)
            throw new ObjectDisposedException(nameof(OperationContext));

        lock (_lock)
        {
            _savedTensors[key] = tensor;
        }
    }

    /// <summary>
    /// Retrieves a saved tensor by key.
    /// </summary>
    /// <typeparam name="T">The type of the saved tensor/object.</typeparam>
    /// <param name="key">The key of the tensor to retrieve.</param>
    /// <returns>The saved tensor/object.</returns>
    public T GetSavedTensor<T>(string key)
    {
        if (string.IsNullOrEmpty(key))
            throw new ArgumentException("Key cannot be null or empty", nameof(key));

        if (_disposed)
            throw new ObjectDisposedException(nameof(OperationContext));

        lock (_lock)
        {
            if (!_savedTensors.TryGetValue(key, out var value))
                throw new KeyNotFoundException($"No tensor saved with key '{key}'");

            if (value is T typedValue)
                return typedValue;

            throw new InvalidCastException($"Saved tensor with key '{key}' is not of type {typeof(T).Name}");
        }
    }

    /// <summary>
    /// Checks if a tensor has been saved with the given key.
    /// </summary>
    /// <param name="key">The key to check.</param>
    /// <returns>True if the key exists, false otherwise.</returns>
    public bool HasSavedTensor(string key)
    {
        if (string.IsNullOrEmpty(key))
            return false;

        lock (_lock)
        {
            return _savedTensors.ContainsKey(key);
        }
    }

    /// <summary>
    /// Clears all saved tensors. This should be called after the backward pass
    /// to free memory.
    /// </summary>
    public void ClearSavedTensors()
    {
        lock (_lock)
        {
            _savedTensors.Clear();
        }
    }

    /// <summary>
    /// Gets the count of saved tensors.
    /// </summary>
    public int SavedTensorCount
    {
        get
        {
            lock (_lock)
            {
                return _savedTensors.Count;
            }
        }
    }
}
