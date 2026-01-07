namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Circular buffer implementation for storing a fixed number of items.
/// </summary>
/// <typeparam name="T">Type of items stored in the buffer.</typeparam>
public class CircularBuffer<T> : IEnumerable<T>
{
    private readonly T[] _buffer;
    private int _head;
    private int _tail;
    private int _count;

    /// <summary>
    /// Gets the number of items in the buffer.
    /// </summary>
    public int Count => _count;

    /// <summary>
    /// Gets the capacity of the buffer.
    /// </summary>
    public int Capacity => _buffer.Length;

    /// <summary>
    /// Creates a new circular buffer with the specified capacity.
    /// </summary>
    /// <param name="capacity">Maximum number of items the buffer can hold.</param>
    public CircularBuffer(int capacity)
    {
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be positive");

        _buffer = new T[capacity];
        _head = 0;
        _tail = 0;
        _count = 0;
    }

    /// <summary>
    /// Adds an item to the buffer, overwriting the oldest item if full.
    /// </summary>
    /// <param name="item">Item to add.</param>
    public void Add(T item)
    {
        _buffer[_tail] = item;
        _tail = (_tail + 1) % _buffer.Length;

        if (_count < _buffer.Length)
        {
            _count++;
        }
        else
        {
            _head = (_head + 1) % _buffer.Length;
        }
    }

    /// <summary>
    /// Gets an enumerator for the buffer.
    /// </summary>
    public IEnumerator<T> GetEnumerator()
    {
        for (int i = 0; i < _count; i++)
        {
            yield return _buffer[(_head + i) % _buffer.Length];
        }
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>
    /// Clears all items from the buffer.
    /// </summary>
    public void Clear()
    {
        _head = 0;
        _tail = 0;
        _count = 0;
    }
}
