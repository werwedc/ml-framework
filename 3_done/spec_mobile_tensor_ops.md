# Spec: Lightweight Tensor Operations

## Overview
Implement lightweight tensor operations optimized for mobile/edge devices without gradient computation.

## Requirements
- No autograd or gradient tracking
- ARM NEON/SVE vectorization hints
- Support for FP32, FP16, Int8, Int16
- In-place operations where safe
- Minimal memory allocations during ops

## Classes to Implement

### 1. `Tensor` Class
```csharp
public sealed class Tensor : ITensor
{
    private IntPtr _data;
    private int[] _shape;
    private DataType _dataType;
    private bool _disposed;

    public Tensor(int[] shape, DataType dataType);
    public Tensor(int[] shape, DataType dataType, IntPtr data, bool ownData);
    ~Tensor();

    public int[] Shape => _shape;
    public DataType DataType => _dataType;
    public long Size => _shape.Aggregate(1L, (a, b) => a * b);
    public long ByteCount => Size * GetDataTypeSize(_dataType);
    public IntPtr DataPointer => _data;

    // Data access
    public T GetData<T>(params int[] indices);
    public T[] ToArray<T>();

    // In-place operations (safe ones)
    public void AddScalar<T>(T scalar);
    public void MultiplyScalar<T>(T scalar);
    public void Clamp<T>(T min, T max);

    // Static factory methods
    public static Tensor Zeros(int[] shape, DataType dataType);
    public static Tensor Ones(int[] shape, DataType dataType);
    public static Tensor FromArray<T>(T[] data, int[] shape);
    public static Tensor Empty(int[] shape, DataType dataType);

    public void Dispose();
    private void Dispose(bool disposing);

    private static int GetDataTypeSize(DataType type);
}
```

### 2. `TensorOperations` Static Class
```csharp
public static class TensorOperations
{
    // Arithmetic operations
    public static Tensor Add(Tensor a, Tensor b);
    public static Tensor Subtract(Tensor a, Tensor b);
    public static Tensor Multiply(Tensor a, Tensor b);
    public static Tensor Divide(Tensor a, Tensor b);

    // Unary operations
    public static Tensor Abs(Tensor input);
    public static Tensor Sqrt(Tensor input);
    public static Tensor Square(Tensor input);
    public static Tensor Log(Tensor input);
    public static Tensor Exp(Tensor input);
    public static Tensor Relu(Tensor input);

    // Reduction operations
    public static Tensor Sum(Tensor input, int axis = -1, bool keepDim = false);
    public static Tensor Mean(Tensor input, int axis = -1, bool keepDim = false);
    public static Tensor Max(Tensor input, int axis = -1, bool keepDim = false);
    public static Tensor Min(Tensor input, int axis = -1, bool keepDim = false);

    // Matrix operations
    public static Tensor MatMul(Tensor a, Tensor b);
    public static Tensor Transpose(Tensor input);

    // Shape operations
    public static Tensor Reshape(Tensor input, int[] newShape);
    public static Tensor Flatten(Tensor input, int startDim = 0, int endDim = -1);
    public static Tensor Expand(Tensor input, int[] newShape);
    public static Tensor Squeeze(Tensor input, int[] dims = null);
    public static Tensor Unsqueeze(Tensor input, int dim);

    // Concatenation and stacking
    public static Tensor Concat(Tensor[] tensors, int axis);
    public static Tensor Stack(Tensor[] tensors, int axis);

    // Comparison operations
    public static Tensor Equal(Tensor a, Tensor b);
    public static Tensor Greater(Tensor a, Tensor b);
    public static Tensor Less(Tensor a, Tensor b);

    // Type conversions
    public static Tensor Cast(Tensor input, DataType targetDataType);

    // Memory operations
    public static Tensor Copy(Tensor input);
    public static void MemCpy(Tensor dst, Tensor src);
}
```

### 3. `ARMVectorization` Helper Class
```csharp
internal static class ARMVectorization
{
    // Vectorized implementations for ARM NEON/SVE
    internal static void VectorizedAdd(IntPtr dst, IntPtr src1, IntPtr src2, long count);
    internal static void VectorizedMultiply(IntPtr dst, IntPtr src1, IntPtr src2, long count);
    internal static void VectorizedRelu(IntPtr dst, IntPtr src, long count);
    internal static void VectorizedSigmoid(IntPtr dst, IntPtr src, long count);

    // Fallback implementations for non-ARM platforms
    private static void ScalarAdd(IntPtr dst, IntPtr src1, IntPtr src2, long count);
    private static void ScalarMultiply(IntPtr dst, IntPtr src1, IntPtr src2, long count);
}
```

### 4. `TensorFactory` Class
```csharp
public class TensorFactory
{
    private readonly IMemoryPool _memoryPool;

    public TensorFactory(IMemoryPool memoryPool);
    public TensorFactory() : this(new DefaultMemoryPool()) { }

    // Factory methods that use memory pool
    public Tensor CreateTensor(int[] shape, DataType dataType);
    public Tensor CreateTensor<T>(T[] data, int[] shape);
    public Tensor CreateView(IntPtr data, int[] shape, DataType dataType);
}
```

## Implementation Notes

### Memory Management
- Use pre-allocated memory from TensorFactory's memory pool
- Implement RAII pattern with Dispose
- Support both owned and unowned data pointers

### Vectorization Strategy
- Detect ARM NEON/SVE at runtime
- Provide scalar fallbacks for non-ARM platforms
- Use `RuntimeInformation.IsOSPlatform(OSPlatform.Create("ARM"))`
- Mark vectorized methods with `[MethodImpl(MethodImplOptions.AggressiveInlining)]`

### Quantization Support
- Int8 operations use saturated arithmetic
- Int16 operations for wider dynamic range
- FP16 operations use half-precision floats

## File Structure
```
src/MobileRuntime/Tensors/
├── Tensor.cs
├── TensorOperations.cs
├── TensorFactory.cs
└── ARMVectorization.cs
```

## Success Criteria
- All operations compile and run
- Vectorization works on ARM platforms
- Fallback works on x86 platforms
- No memory leaks in stress tests
- Operations are allocation-free (except result tensors)
- Supports FP32, FP16, Int8, Int16 data types

## Dependencies
- spec_mobile_runtime_core (interfaces)
- spec_mobile_memory_pool (memory pool)

## Testing Requirements
- Unit tests for each operation
- Comparison against reference implementations
- Memory leak detection
- Vectorization verification on ARM platforms

## Performance Targets
- Add/Multiply: < 1ms per 1M elements on modern ARM CPU
- MatMul: < 10ms for 1000x1000 matrix
- Zero allocations during operation execution
