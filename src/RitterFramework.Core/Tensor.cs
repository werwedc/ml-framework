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

    public int[] Shape => _shape;
    public int Size => _data.Length;
    public int Dimensions => _shape.Length;

    // Internal access to data for gradient operations
    public float[] Data => _data;

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