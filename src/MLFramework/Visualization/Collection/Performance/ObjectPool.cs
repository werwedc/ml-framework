using System.Collections.Concurrent;

namespace MachineLearning.Visualization.Collection.Performance;

/// <summary>
/// Generic object pool for reusing objects to reduce allocations
/// </summary>
/// <typeparam name="T">The type of object to pool</typeparam>
public class ObjectPool<T> where T : class, new()
{
    private readonly ConcurrentBag<T> _objects;
    private readonly int _capacity;
    private readonly Action<T>? _resetAction;
    private long _totalAllocated;
    private long _totalReused;

    /// <summary>
    /// Gets the current number of objects in the pool
    /// </summary>
    public int Count => _objects.Count;

    /// <summary>
    /// Gets the total number of objects allocated from the pool
    /// </summary>
    public long TotalAllocated => Interlocked.Read(ref _totalAllocated);

    /// <summary>
    /// Gets the total number of objects reused from the pool
    /// </summary>
    public long TotalReused => Interlocked.Read(ref _totalReused);

    /// <summary>
    /// Creates a new object pool with the specified capacity
    /// </summary>
    /// <param name="capacity">Maximum number of objects to keep in the pool</param>
    /// <param name="resetAction">Optional action to reset objects when they are returned to the pool</param>
    public ObjectPool(int capacity, Action<T>? resetAction = null)
    {
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be positive");

        _capacity = capacity;
        _resetAction = resetAction;
        _objects = new ConcurrentBag<T>();
    }

    /// <summary>
    /// Gets an object from the pool, creating a new one if necessary
    /// </summary>
    /// <returns>An object from the pool</returns>
    public T Rent()
    {
        if (_objects.TryTake(out var obj))
        {
            Interlocked.Increment(ref _totalReused);
            return obj;
        }

        Interlocked.Increment(ref _totalAllocated);
        return new T();
    }

    /// <summary>
    /// Returns an object to the pool
    /// </summary>
    /// <param name="obj">The object to return</param>
    public void Return(T obj)
    {
        if (obj == null)
            throw new ArgumentNullException(nameof(obj));

        // Reset the object if a reset action was provided
        _resetAction?.Invoke(obj);

        // Only return to pool if we're under capacity
        if (_objects.Count < _capacity)
        {
            _objects.Add(obj);
        }
    }

    /// <summary>
    /// Clears all objects from the pool
    /// </summary>
    public void Clear()
    {
        while (_objects.TryTake(out _))
        {
            // Dispose or cleanup could be added here
        }
    }

    /// <summary>
    /// Gets the reuse rate (percentage of objects that were reused vs allocated)
    /// </summary>
    public double GetReuseRate()
    {
        long allocated = TotalAllocated;
        long reused = TotalReused;

        if (allocated == 0)
            return 0.0;

        return (double)reused / (allocated + reused) * 100;
    }
}
