using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Placeholder for symbolic shape representation.
/// TODO: Implement fully when spec_symbolic_shape.md is processed.
/// </summary>
public class SymbolicShape
{
    public int Rank { get; }
    public bool IsKnown { get; set; }

    public SymbolicShape(int rank)
    {
        Rank = rank;
        IsKnown = false;
    }

    public int[] ToConcrete()
    {
        throw new NotImplementedException("SymbolicShape not yet fully implemented");
    }
}

/// <summary>
/// Manages a dynamic buffer for accumulating gradients with variable sizes.
/// Supports efficient accumulation with buffer resizing and slicing operations.
/// </summary>
public class AccumulationBufferDynamic
{
    private Tensor? _buffer;
    private readonly SymbolicShape _bufferShape;
    private int _currentSize;
    private readonly int _maxSize;

    /// <summary>
    /// Gets the current buffer tensor.
    /// </summary>
    public Tensor? Buffer => _buffer;

    /// <summary>
    /// Gets the symbolic shape of the buffer.
    /// </summary>
    public SymbolicShape BufferShape => _bufferShape;

    /// <summary>
    /// Gets the current size (number of elements) in the buffer.
    /// </summary>
    public int CurrentSize => _currentSize;

    /// <summary>
    /// Gets the maximum capacity of the buffer.
    /// </summary>
    public int MaxSize => _maxSize;

    /// <summary>
    /// Initializes a new instance of the AccumulationBufferDynamic class.
    /// </summary>
    /// <param name="bufferShape">The symbolic shape of the buffer.</param>
    /// <param name="maxSize">The maximum size of the buffer.</param>
    /// <exception cref="ArgumentNullException">Thrown when bufferShape is null.</exception>
    /// <exception cref="ArgumentException">Thrown when maxSize is less than 1.</exception>
    public AccumulationBufferDynamic(SymbolicShape bufferShape, int maxSize)
    {
        _bufferShape = bufferShape ?? throw new ArgumentNullException(nameof(bufferShape));
        if (maxSize < 1)
            throw new ArgumentException("Max size must be at least 1", nameof(maxSize));

        _maxSize = maxSize;
        _currentSize = 0;
        _buffer = null;
    }

    /// <summary>
    /// Initializes a new instance of the AccumulationBufferDynamic class with concrete shape.
    /// </summary>
    /// <param name="shape">The concrete shape of the buffer (as int array).</param>
    /// <param name="maxSize">The maximum size of the buffer.</param>
    public AccumulationBufferDynamic(int[] shape, int maxSize) : this(new SymbolicShape(shape.Length), maxSize)
    {
        // Initialize buffer with concrete shape
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape cannot be null or empty", nameof(shape));

        _bufferShape.IsKnown = true;
        _buffer = Tensor.Zeros(shape);
    }

    /// <summary>
    /// Accumulates a gradient into the buffer at the specified position.
    /// </summary>
    /// <param name="gradient">The gradient tensor to accumulate.</param>
    /// <param name="startIdx">The starting index for accumulation.</param>
    /// <param name="count">The number of elements to accumulate.</param>
    /// <exception cref="ArgumentNullException">Thrown when gradient is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when buffer is not initialized or indices are invalid.</exception>
    public void Accumulate(Tensor gradient, int startIdx, int count)
    {
        if (gradient == null)
            throw new ArgumentNullException(nameof(gradient));

        if (_buffer == null)
            throw new InvalidOperationException("Buffer is not initialized");

        if (startIdx < 0 || startIdx >= _buffer.Data.Length)
            throw new InvalidOperationException("Start index out of bounds");

        if (count < 0 || startIdx + count > _buffer.Data.Length)
            throw new InvalidOperationException("Count exceeds buffer bounds");

        if (gradient.Data.Length < count)
            throw new InvalidOperationException("Gradient size smaller than count");

        // Accumulate gradient values
        for (int i = 0; i < count; i++)
        {
            _buffer.Data[startIdx + i] += gradient.Data[i];
        }

        _currentSize = Math.Max(_currentSize, startIdx + count);
    }

    /// <summary>
    /// Resizes the buffer to a new size.
    /// Preserves existing data within the new size limit.
    /// </summary>
    /// <param name="newSize">The new buffer size.</param>
    /// <exception cref="ArgumentException">Thrown when newSize is less than 1 or exceeds max size.</exception>
    public void Resize(int newSize)
    {
        if (newSize < 1)
            throw new ArgumentException("New size must be at least 1", nameof(newSize));

        if (newSize > _maxSize)
            throw new ArgumentException("New size exceeds maximum buffer size", nameof(newSize));

        if (_buffer == null)
            throw new InvalidOperationException("Buffer is not initialized");

        // Create new buffer with new size (only supports 1D for now)
        // For multi-dimensional, we'd need to compute the new shape
        var oldData = _buffer.Data;
        var newData = new float[newSize];

        // Copy existing data
        int copyCount = Math.Min(oldData.Length, newData.Length);
        Array.Copy(oldData, newData, copyCount);

        // Create new tensor with updated shape
        _buffer = new Tensor(newData, newSize > 0 ? new[] { newSize } : new int[0]);
    }

    /// <summary>
    /// Gets a slice of the buffer from the specified position.
    /// </summary>
    /// <param name="startIdx">The starting index.</param>
    /// <param name="count">The number of elements to retrieve.</param>
    /// <returns>A new tensor containing the slice.</returns>
    /// <exception cref="InvalidOperationException">Thrown when buffer is not initialized or indices are invalid.</exception>
    public Tensor GetSlice(int startIdx, int count)
    {
        if (_buffer == null)
            throw new InvalidOperationException("Buffer is not initialized");

        if (startIdx < 0 || startIdx >= _buffer.Data.Length)
            throw new InvalidOperationException("Start index out of bounds");

        if (count < 0 || startIdx + count > _buffer.Data.Length)
            throw new InvalidOperationException("Count exceeds buffer bounds");

        var sliceData = new float[count];
        Array.Copy(_buffer.Data, startIdx, sliceData, 0, count);

        return new Tensor(sliceData, new[] { count });
    }

    /// <summary>
    /// Gets the full buffer content.
    /// </summary>
    /// <returns>A new tensor containing the full buffer.</returns>
    /// <exception cref="InvalidOperationException">Thrown when buffer is not initialized.</exception>
    public Tensor GetFull()
    {
        if (_buffer == null)
            throw new InvalidOperationException("Buffer is not initialized");

        // Return a copy
        var bufferCopy = new float[_buffer.Data.Length];
        Array.Copy(_buffer.Data, bufferCopy, _buffer.Data.Length);

        return new Tensor(bufferCopy, _buffer.Shape);
    }

    /// <summary>
    /// Clears the buffer content.
    /// </summary>
    public void Clear()
    {
        if (_buffer != null)
        {
            Array.Clear(_buffer.Data, 0, _buffer.Data.Length);
        }
        _currentSize = 0;
    }

    /// <summary>
    /// Gets a value indicating whether the buffer is full.
    /// </summary>
    public bool IsFull => _currentSize >= _maxSize;

    /// <summary>
    /// Gets the available capacity in the buffer.
    /// </summary>
    public int AvailableCapacity => _maxSize - _currentSize;
}
