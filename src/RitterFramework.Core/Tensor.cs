namespace RitterFramework.Core.Tensor;

public class Tensor
{
    private float[] _data;
    private int[] _shape;
    private int[] _strides;

    public Tensor? Gradient { get; set; }
    public bool RequiresGrad { get; set; }
    public List<Tensor>? Parents { get; set; }
    public Action<Tensor>? BackwardFn { get; set; }
    public DataType Dtype { get; set; }
    public Guid Id { get; private set; } = Guid.NewGuid();

    /// <summary>
    /// Gets the backward function associated with this tensor.
    /// </summary>
    /// <returns>The backward function, or null if not set.</returns>
    public Action<Tensor>? GetGradFn()
    {
        return BackwardFn;
    }

    /// <summary>
    /// Sets the backward function for this tensor.
    /// </summary>
    /// <param name="fn">The backward function to set.</param>
    public void SetGradFn(Action<Tensor>? fn)
    {
        BackwardFn = fn;
    }

    // Track the operation that created this tensor (optional)
    public string? SourceOperation { get; set; }

    // Track the layer/module that created this tensor (optional)
    public string? SourceLayer { get; set; }

    public int[] Shape => _shape;
    public int Size => _data.Length;
    public int Dimensions => _shape.Length;

    // Internal access to data for gradient operations
    public float[] Data => _data;

    /// <summary>
    /// Gets the shape as a formatted string.
    /// </summary>
    /// <returns>Shape formatted as [dim1, dim2, ...]</returns>
    public string GetShapeString()
    {
        return $"[{string.Join(", ", _shape)}]";
    }

    /// <summary>
    /// Gets the number of dimensions (rank) of the tensor.
    /// </summary>
    /// <returns>The rank of the tensor.</returns>
    public int GetRank()
    {
        return _shape.Length;
    }

    /// <summary>
    /// Gets the size of a specific dimension.
    /// </summary>
    /// <param name="index">The dimension index.</param>
    /// <returns>The size of the dimension.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of range.</exception>
    public int GetDimension(int index)
    {
        if (index < 0 || index >= _shape.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(index),
                $"Dimension index {index} is out of range. Tensor has {GetRank()} dimensions.");
        }
        return _shape[index];
    }

    /// <summary>
    /// Checks if this tensor has the same shape as another tensor.
    /// </summary>
    /// <param name="other">The other tensor to compare with.</param>
    /// <returns>True if shapes match, false otherwise.</returns>
    public bool HasSameShape(Tensor other)
    {
        if (other == null)
        {
            return false;
        }

        return _shape.SequenceEqual(other._shape);
    }

    public Tensor(float[] data, int[] shape, bool requiresGrad = false, DataType dtype = DataType.Float32)
    {
        _data = data;
        _shape = shape;
        _strides = ComputeStrides(shape);

        RequiresGrad = requiresGrad;
        Dtype = dtype;

        if (requiresGrad)
        {
            Gradient = Zeros(shape);
        }
    }

    public static Tensor Zeros(int[] shape, DataType dtype = DataType.Float32)
    {
        if(shape.Length < 1) throw new ArgumentOutOfRangeException(nameof(shape));

        var length =  1;
        foreach(var dimension in shape) length *= dimension;

        return new Tensor(new float[length], shape, false, dtype);
    }

    public static Tensor Ones(int[] shape, DataType dtype = DataType.Float32)
    {
        if(shape.Length < 1) throw new ArgumentOutOfRangeException(nameof(shape));

        var length =  1;
        foreach(var dimension in shape) length *= dimension;

        var contents = new float[length];
        for (int i = 0; i < length; i++)
        {
            contents[i] = 1;
        }

        return new Tensor(contents, shape, false, dtype);
    }

    public static Tensor FromArray(float[] data, DataType dtype = DataType.Float32)
    {
        if (data == null) throw new ArgumentNullException(nameof(data));
        if (data.Length == 0) throw new ArgumentException("Data array cannot be empty", nameof(data));

        var shape = new int[] { data.Length };
        return new Tensor(data, shape, false, dtype);
    }
    
    public float this[int[] indices]
    {
        get =>
            _data[GetFlatIndex(indices)];
        set =>
            _data[GetFlatIndex(indices)] = value;
    }

    public static Tensor operator +(Tensor tensorA, Tensor tensorB)
    {
        if (!tensorA._shape.SequenceEqual(tensorB._shape))
            throw new ArgumentException("Shapes must match for addition");
        
        var newContents = tensorA._data.Zip(tensorB._data, (a, b) => a + b).ToArray();
        var newTensor = new Tensor(newContents, tensorA._shape, tensorA.RequiresGrad || tensorB.RequiresGrad);
            
        if (tensorA.RequiresGrad || tensorB.RequiresGrad)
        {  
            newTensor.Parents = new List<Tensor>{tensorA, tensorB};
            newTensor.BackwardFn = gradOutput =>
            {
                if (tensorA.RequiresGrad)
                {
                    tensorA.Gradient!._data = tensorA._data.Zip(gradOutput._data, (a, g) => a + g).ToArray();
                    
                    tensorA.Backward(gradOutput);
                }

                if (tensorB.RequiresGrad)
                {
                    tensorB.Gradient!._data = tensorB._data.Zip(gradOutput._data, (b, g) => b + g).ToArray();
                    
                    tensorB.Backward(gradOutput);
                }
            };
        }
            
        return newTensor;
    }
    
    public static Tensor operator *(Tensor tensor, float scalar)
    {
        var newContents = tensor._data.Select(a => a * scalar).ToArray();
        var newTensor = new Tensor(newContents, tensor._shape, tensor.RequiresGrad);
            
        if (tensor.RequiresGrad)
        {  
            newTensor.Parents = new List<Tensor>{tensor};
            newTensor.BackwardFn = gradOutput =>
            {
                tensor.Gradient!._data = tensor._data.Select(a => a * scalar).ToArray();
                
                tensor.Backward(gradOutput);
            };
        }
            
        return newTensor;
    }

    /// <summary>
    /// Accumulates gradients from a single backward pass.
    /// Used by custom functions to add gradients to existing gradient tensor.
    /// </summary>
    /// <param name="grad">The gradient tensor to accumulate.</param>
    public void AccumulateGrad(Tensor grad)
    {
        if (grad == null)
            throw new ArgumentNullException(nameof(grad));

        if (Gradient == null)
            Gradient = Zeros(Shape);

        for (int i = 0; i < Size; i++)
            Gradient._data[i] += grad._data[i];
    }

    public void Backward(Tensor? gradOutput = null)
    {
        if (gradOutput == null)
        {
            if (Size != 1)
                throw new ArgumentException("Gradient must be provided for non-scalar tensors");

            gradOutput = Ones(Shape);
        }

        if(Gradient == null) Gradient = Zeros(Shape);

        // TODO: Increment global pass ID
        for (int i = 0; i < Size; i++)
            Gradient._data[i] += gradOutput._data[i];

        BackwardFn?.Invoke(gradOutput);
    }

    /// <summary>
    /// Creates a deep copy of this tensor.
    /// </summary>
    public Tensor Clone()
    {
        var newData = new float[_data.Length];
        Array.Copy(_data, newData, _data.Length);

        var newShape = new int[_shape.Length];
        Array.Copy(_shape, newShape, _shape.Length);

        return new Tensor(newData, newShape, RequiresGrad, Dtype);
    }

    /// <summary>
    /// Reshapes the tensor to the specified shape.
    /// </summary>
    public Tensor Reshape(int[] newShape)
    {
        long newSize = 1;
        foreach (var dim in newShape)
            newSize *= dim;

        if (newSize != _data.Length)
            throw new ArgumentException($"Cannot reshape tensor of size {_data.Length} to shape [{string.Join(", ", newShape)}]");

        return new Tensor(_data, newShape, RequiresGrad, Dtype);
    }

    /// <summary>
    /// Transposes a 2D tensor.
    /// </summary>
    public Tensor Transpose()
    {
        if (Dimensions != 2)
            throw new InvalidOperationException("Transpose is only supported for 2D tensors");

        int rows = _shape[0];
        int cols = _shape[1];
        var newData = new float[_data.Length];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                newData[j * rows + i] = _data[i * cols + j];
            }
        }

        return new Tensor(newData, new[] { cols, rows }, RequiresGrad, Dtype);
    }

    /// <summary>
    /// Copies data from another tensor into this tensor.
    /// </summary>
    public void CopyFrom(Tensor other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        if (Size != other.Size)
            throw new ArgumentException("Tensor sizes must match for copy operation");

        Array.Copy(other.Data, _data, _data.Length);
    }
    
    private int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        var stride = 1;

        for (var i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        
        return strides;
    }

    private int GetFlatIndex(int[] indices) =>
        indices.Zip(_strides, (idx, stride) => new { idx, stride })
            .Select((item, i) =>
            {
                if (item.idx < 0 || item.idx >= _shape[i])
                    throw new IndexOutOfRangeException($"Index {item.idx} at dimension {i} is out of bounds.");

                return item.idx * item.stride;
            }).Sum();
}