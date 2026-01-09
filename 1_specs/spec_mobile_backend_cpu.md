# Spec: CPU Backend Implementation

## Overview
Implement CPU execution backend with ARM NEON/SVE optimizations for mobile runtime.

## Requirements
- ARM NEON/SVE vectorization
- Static graph execution (no dynamic shapes)
- Operator fusion support
- Loop unrolling and vectorization
- Platform detection and fallback
- Multi-threading for large tensors

## Classes to Implement

### 1. `ICpuBackend` Interface
```csharp
public interface ICpuBackend
{
    string Name { get; }
    BackendCapabilities Capabilities { get; }

    // Operator execution
    ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters);

    // Batch execution
    ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry);

    // Platform info
    CpuInfo GetCpuInfo();

    // Optimization
    void EnableVectorization(bool enable);
    void EnableMultiThreading(bool enable, int maxThreads = 0);
}

public class BackendCapabilities
{
    public bool SupportsNeon { get; set; }
    public bool SupportsSve { get; set; }
    public bool SupportsAvx { get; set; }
    public bool SupportsAvx2 { get; set; }
    public bool SupportsAvx512 { get; set; }
    public int MaxThreads { get; set; }
    public long CacheLineSize { get; set; }
}

public class CpuInfo
{
    public string Vendor { get; set; }
    public string Model { get; set; }
    public int CoreCount { get; set; }
    public int ThreadCount { get; set; }
    public int FrequencyMHz { get; set; }
    public BackendCapabilities Capabilities { get; set; }
}
```

### 2. `CpuBackend` Class
```csharp
public sealed class CpuBackend : ICpuBackend
{
    private readonly IMemoryPool _memoryPool;
    private readonly ITensorFactory _tensorFactory;
    private readonly CpuInfo _cpuInfo;
    private readonly Dictionary<OperatorType, IOperatorExecutor> _executors;
    private bool _vectorizationEnabled;
    private bool _multiThreadingEnabled;
    private int _maxThreads;

    public CpuBackend(IMemoryPool memoryPool, ITensorFactory tensorFactory);

    public string Name => "CPU";
    public BackendCapabilities Capabilities => _cpuInfo.Capabilities;

    public ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters);
    public ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry);

    public CpuInfo GetCpuInfo() => _cpuInfo;

    public void EnableVectorization(bool enable) => _vectorizationEnabled = enable;
    public void EnableMultiThreading(bool enable, int maxThreads = 0);

    private IOperatorExecutor GetExecutor(OperatorType type);
    private void InitializeExecutors();
    private void DetectCpuCapabilities();
}
```

### 3. `IOperatorExecutor` Interface
```csharp
public interface IOperatorExecutor
{
    OperatorType OperatorType { get; }

    ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters);

    bool CanFuseWith(IOperatorExecutor other);
    ITensor ExecuteFused(IOperatorExecutor[] executors, ITensor[][] inputs, Dictionary<string, object>[] parameters);
}
```

### 4. Operator Executors

#### Conv2DExecutor
```csharp
public sealed class Conv2DExecutor : IOperatorExecutor
{
    public OperatorType OperatorType => OperatorType.Conv2D;

    private readonly CpuBackend _backend;

    public ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters);

    public bool CanFuseWith(IOperatorExecutor other)
    {
        // Can fuse with BatchNorm, Relu, Sigmoid
        return other is BatchNormExecutor || other is ReluExecutor;
    }

    public ITensor ExecuteFused(IOperatorExecutor[] executors, ITensor[][] inputs, Dictionary<string, object>[] parameters);

    private ITensor ExecuteConv2D(ITensor input, ITensor weight, ITensor bias, Conv2DParams params);
    private ITensor ExecuteIm2Col(ITensor input, int kernelSize, int stride, int padding);
    private ITensor ExecuteGemm(ITensor im2Col, ITensor weight, ITensor bias, int outputChannels);
}

public class Conv2DParams
{
    public int[] KernelSize { get; set; }
    public int[] Stride { get; set; }
    public int[] Padding { get; set; }
    public int[] Dilation { get; set; }
    public int Groups { get; set; }
}
```

#### ReluExecutor
```csharp
public sealed class ReluExecutor : IOperatorExecutor
{
    public OperatorType OperatorType => OperatorType.Relu;

    private readonly CpuBackend _backend;

    public ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters)
    {
        var input = inputs[0];
        return TensorOperations.Relu(input);
    }

    public bool CanFuseWith(IOperatorExecutor other)
    {
        // Can be fused after Conv2D, FullyConnected
        return other is Conv2DExecutor || other is FullyConnectedExecutor;
    }

    public ITensor ExecuteFused(IOperatorExecutor[] executors, ITensor[][] inputs, Dictionary<string, object>[] parameters)
    {
        // If fused with Conv2D, apply ReLU in-place after convolution
        if (executors[0] is Conv2DExecutor conv)
        {
            return conv.ExecuteFused(new[] { this }, inputs, parameters);
        }
        return Execute(inputs[0], parameters);
    }
}
```

#### MaxPool2DExecutor
```csharp
public sealed class MaxPool2DExecutor : IOperatorExecutor
{
    public OperatorType OperatorType => OperatorType.MaxPool2D;

    public ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters);

    private ITensor ExecuteMaxPool2D(ITensor input, Pool2DParams params);

    private ITensor NaiveMaxPool2D(ITensor input, Pool2DParams params);
    private ITensor VectorizedMaxPool2D(ITensor input, Pool2DParams params);
}

public class Pool2DParams
{
    public int[] KernelSize { get; set; }
    public int[] Stride { get; set; }
    public int[] Padding { get; set; }
    public bool CountIncludePad { get; set; }
}
```

#### FullyConnectedExecutor
```csharp
public sealed class FullyConnectedExecutor : IOperatorExecutor
{
    public OperatorType OperatorType => OperatorType.FullyConnected;

    public ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters);

    private ITensor ExecuteFullyConnected(ITensor input, ITensor weight, ITensor bias);
}
```

#### AddExecutor, MultiplyExecutor, etc.
```csharp
// Similar structure for Add, Subtract, Multiply, Divide, Concat, etc.
public sealed class AddExecutor : IOperatorExecutor
{
    public OperatorType OperatorType => OperatorType.Add;

    public ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters)
    {
        return TensorOperations.Add(inputs[0], inputs[1]);
    }

    public bool CanFuseWith(IOperatorExecutor other) => false;
    public ITensor ExecuteFused(IOperatorExecutor[] executors, ITensor[][] inputs, Dictionary<string, object>[] parameters)
        => throw new NotSupportedException();
}
```

### 5. `CpuBackendFactory` Class
```csharp
public static class CpuBackendFactory
{
    public static ICpuBackend CreateDefault(IMemoryPool memoryPool, ITensorFactory tensorFactory);
    public static ICpuBackend CreateWithNeonOptimization(IMemoryPool memoryPool, ITensorFactory tensorFactory);
    public static ICpuBackend CreateForX86(IMemoryPool memoryPool, ITensorFactory tensorFactory);
}
```

### 6. `CpuVectorization` Utility Class
```csharp
internal static class CpuVectorization
{
    // NEON intrinsics (ARM64)
    internal static unsafe void NeonAdd(float* dst, float* src1, float* src2, int count);
    internal static unsafe void NeonMultiply(float* dst, float* src1, float* src2, int count);
    internal static unsafe void NeonRelu(float* dst, float* src, int count);
    internal static unsafe void NeonSigmoid(float* dst, float* src, int count);

    // AVX intrinsics (x86)
    internal static unsafe void AvxAdd(float* dst, float* src1, float* src2, int count);
    internal static unsafe void AvxMultiply(float* dst, float* src1, float* src2, int count);

    // Scalar fallbacks
    private static unsafe void ScalarAdd(float* dst, float* src1, float* src2, int count);
    private static unsafe void ScalarMultiply(float* dst, float* src1, float* src2, int count);
}
```

## Implementation Notes

### Vectorization Strategy
- Detect ARM NEON/SVE at runtime
- Use vectorized path when count >= 8 (for NEON 128-bit)
- Use scalar fallback for remaining elements
- Mark vectorized methods with `[MethodImpl(MethodImplOptions.AggressiveInlining)]`

### Multi-threading
- Use `Parallel.For` for large tensors (> 64KB)
- Partition by cache lines (64 bytes)
- Use `Environment.ProcessorCount` as default max threads
- Disable multi-threading for small tensors (< 4KB)

### Memory Access Patterns
- Prefetch data for better cache utilization
- Process data in cache-friendly order (row-major)
- Use `MemoryMarshal.Cast<T, byte>` for raw memory access

### Operator Fusion
- Fuse Conv2D + BatchNorm
- Fuse Conv2D + Relu
- Fuse FullyConnected + Relu
- Fuse multiple activation layers

## File Structure
```
src/MobileRuntime/Backends/Cpu/
├── CpuBackend.cs
├── CpuBackendFactory.cs
├── Interfaces/
│   ├── ICpuBackend.cs
│   └── IOperatorExecutor.cs
├── Executors/
│   ├── Conv2DExecutor.cs
│   ├── ReluExecutor.cs
│   ├── MaxPool2DExecutor.cs
│   ├── FullyConnectedExecutor.cs
│   ├── AddExecutor.cs
│   ├── MultiplyExecutor.cs
│   ├── ConcatExecutor.cs
│   └── ReshapeExecutor.cs
├── Models/
│   ├── BackendCapabilities.cs
│   ├── CpuInfo.cs
│   ├── Conv2DParams.cs
│   └── Pool2DParams.cs
└── Utils/
    └── CpuVectorization.cs
```

## Success Criteria
- All operators execute correctly
- NEON vectorization detected and used on ARM
- Fallback works on x86 platforms
- Multi-threading speeds up large tensor ops
- Operator fusion reduces kernel launches
- Performance meets targets

## Dependencies
- spec_mobile_runtime_core (interfaces)
- spec_mobile_tensor_ops (Tensor, TensorOperations)
- spec_mobile_memory_pool (IMemoryPool)

## Testing Requirements
- Unit tests for each operator
- Comparison against reference implementations
- Vectorization verification on ARM platforms
- Multi-threading correctness tests
- Fusion correctness tests
- Performance benchmarks

## Performance Targets
- Conv2D (3x3, 32 channels, 224x224 input): < 5ms
- FullyConnected (1024 -> 1024): < 1ms
- MaxPool2D (2x2, 224x224): < 1ms
- Element-wise ops (1M elements): < 1ms
- Multi-threading speedup: > 3x on 4+ core CPUs
