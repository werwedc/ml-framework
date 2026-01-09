# Spec: Metal GPU Backend (iOS)

## Overview
Implement Metal-based GPU backend for iOS devices with hardware acceleration for inference.

## Requirements
- Metal 2.0+ support
- Compute shaders for all operators
- MTLBuffer-based memory management
- Multi-command queue support
- Texture-based convolution optimization
- MPS (Metal Performance Shaders) integration fallback

## Classes to Implement

### 1. `IMetalBackend` Interface
```csharp
public interface IMetalBackend
{
    string Name { get; }
    MetalBackendCapabilities Capabilities { get; }

    // Memory management
    IMetalBuffer AllocateBuffer(long size);
    void FreeBuffer(IMetalBuffer buffer);

    // Operator execution
    ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters);

    // Batch execution
    ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry);

    // Compute shaders
    IMetalComputeShader GetComputeShader(OperatorType opType);

    // Synchronization
    void WaitForCompletion();

    // Platform info
    MetalDeviceInfo GetDeviceInfo();
}

public class MetalBackendCapabilities
{
    public bool SupportsMPS { get; set; }
    public bool SupportsUnifiedMemory { get; set; }
    public int MaxThreadsPerThreadgroup { get; set; }
    public int MaxTextureWidth { get; set; }
    public int MaxTextureHeight { get; set; }
    public int MaxBufferLength { get; set; }
}

public class MetalDeviceInfo
{
    public string DeviceName { get; set; }
    public uint FamilyId { get; set; }
    public uint RecommendedMaxWorkingSetSize { get; set; }
    public bool HasUnifiedMemory { get; set; }
    public MetalBackendCapabilities Capabilities { get; set; }
}
```

### 2. `MetalBackend` Class
```csharp
public sealed class MetalBackend : IMetalBackend, IDisposable
{
    private IntPtr _device;
    private IntPtr _commandQueue;
    private readonly Dictionary<OperatorType, IMetalComputeShader> _shaders;
    private readonly List<IMetalBuffer> _allocatedBuffers;
    private readonly ITensorFactory _tensorFactory;

    public MetalBackend(ITensorFactory tensorFactory);
    ~MetalBackend();

    public string Name => "Metal";
    public MetalBackendCapabilities Capabilities { get; private set; }

    public IMetalBuffer AllocateBuffer(long size);
    public void FreeBuffer(IMetalBuffer buffer);

    public ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters);
    public ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry);

    public IMetalComputeShader GetComputeShader(OperatorType opType);
    public void WaitForCompletion();

    public MetalDeviceInfo GetDeviceInfo();

    public void Dispose();
    private void Dispose(bool disposing);

    private void InitializeMetal();
    private void InitializeShaders();
    private void DetectCapabilities();
}
```

### 3. `IMetalBuffer` Interface
```csharp
public interface IMetalBuffer : IDisposable
{
    IntPtr NativeBuffer { get; }
    long Length { get; }
    IntPtr Contents { get; }

    void CopyFrom(IntPtr source, long size);
    void CopyTo(IntPtr destination, long size);
    void DidModifyRange(long location, long length);
}
```

### 4. `MetalBuffer` Class
```csharp
public sealed class MetalBuffer : IMetalBuffer
{
    private IntPtr _buffer;
    private readonly long _length;
    private readonly IntPtr _device;

    public MetalBuffer(IntPtr device, long length);
    ~MetalBuffer();

    public IntPtr NativeBuffer => _buffer;
    public long Length => _length;
    public IntPtr Contents { get; }

    public void CopyFrom(IntPtr source, long size);
    public void CopyTo(IntPtr destination, long size);
    public void DidModifyRange(long location, long length);

    public void Dispose();
    private void Dispose(bool disposing);
}
```

### 5. `IMetalComputeShader` Interface
```csharp
public interface IMetalComputeShader
{
    string Name { get; }
    OperatorType OperatorType { get; }

    void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters);

    void SetBytes(IntPtr buffer, long offset, IntPtr data, long size);
    void SetBuffer(MetalBuffer buffer, int index);
}
```

### 6. Compute Shaders

#### MetalConv2DShader
```csharp
public sealed class MetalConv2DShader : IMetalComputeShader
{
    private readonly MetalBackend _backend;
    private readonly IntPtr _pipelineState;

    public string Name => "Conv2D";
    public OperatorType OperatorType => OperatorType.Conv2D;

    public MetalConv2DShader(MetalBackend backend);

    public void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters);
    public void SetBytes(IntPtr buffer, long offset, IntPtr data, long size);
    public void SetBuffer(MetalBuffer buffer, int index);

    private void InitializePipelineState();
}
```

#### MetalReluShader
```csharp
public sealed class MetalReluShader : IMetalComputeShader
{
    private readonly MetalBackend _backend;
    private readonly IntPtr _pipelineState;

    public string Name => "Relu";
    public OperatorType OperatorType => OperatorType.Relu;

    public MetalReluShader(MetalBackend backend);

    public void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters);
    public void SetBytes(IntPtr buffer, long offset, IntPtr data, long size);
    public void SetBuffer(MetalBuffer buffer, int index);

    private void InitializePipelineState();
}
```

#### MetalMaxPool2DShader
```csharp
public sealed class MetalMaxPool2DShader : IMetalComputeShader
{
    private readonly MetalBackend _backend;
    private readonly IntPtr _pipelineState;

    public string Name => "MaxPool2D";
    public OperatorType OperatorType => OperatorType.MaxPool2D;

    public MetalMaxPool2DShader(MetalBackend backend);

    public void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters);
    public void SetBytes(IntPtr buffer, long offset, IntPtr data, long size);
    public void SetBuffer(MetalBuffer buffer, int index);

    private void InitializePipelineState();
}
```

#### MetalElementWiseShader
```csharp
public sealed class MetalElementWiseShader : IMetalComputeShader
{
    private readonly MetalBackend _backend;
    private readonly IntPtr _pipelineState;
    private readonly OperatorType _opType;

    public string Name => _opType.ToString();
    public OperatorType OperatorType => _opType;

    public MetalElementWiseShader(MetalBackend backend, OperatorType opType);

    public void Dispatch(MetalCommandBuffer commandBuffer, MetalBuffer[] inputs, MetalBuffer[] outputs, Dictionary<string, object> parameters);
    public void SetBytes(IntPtr buffer, long offset, IntPtr data, long size);
    public void SetBuffer(MetalBuffer buffer, int index);

    private void InitializePipelineState();
}
```

### 7. `MetalCommandBuffer` Helper Class
```csharp
public sealed class MetalCommandBuffer : IDisposable
{
    private readonly IntPtr _commandBuffer;
    private readonly List<MetalEncoder> _encoders;
    private readonly MetalBackend _backend;

    public MetalCommandBuffer(MetalBackend backend);

    public MetalComputeCommandEncoder CreateComputeCommandEncoder();
    public MetalBlitCommandEncoder CreateBlitCommandEncoder();

    public void Commit();
    public void WaitUntilCompleted();

    public void Dispose();
}

public sealed class MetalComputeCommandEncoder
{
    private readonly IntPtr _encoder;

    public void SetComputePipelineState(IntPtr pipelineState);
    public void SetBuffer(IntPtr buffer, int index, long offset);
    public void SetBytes(IntPtr bytes, long length, int index);
    public void DispatchThreadgroups(MTLSize threadgroups, MTLSize threadsPerThreadgroup);
    public void EndEncoding();
}

public sealed class MetalBlitCommandEncoder
{
    private readonly IntPtr _encoder;

    public void CopyFromBuffer(IntPtr source, long sourceOffset, IntPtr destination, long destinationOffset, long size);
    public void Synchronize(IntPtr resource);
    public void EndEncoding();
}

public struct MTLSize
{
    public long Width;
    public long Height;
    public long Depth;
}
```

### 8. `MetalBackendFactory` Class
```csharp
public static class MetalBackendFactory
{
    public static IMetalBackend CreateDefault(ITensorFactory tensorFactory);
    public static IMetalBackend CreateWithMPSFallback(ITensorFactory tensorFactory);
    public static bool IsAvailable();
}
```

## Implementation Notes

### Metal Interop
- Use P/Invoke to call Metal APIs
- Link against Metal.framework via native library
- Use `[DllImport("__Internal")]` for iOS calls
- Handle Objective-C ARC manually

### Shader Compilation
- Pre-compile shaders at backend initialization
- Use `.metal` files compiled to `.metallib` via Xcode
- Load compiled shaders from bundle
- Fall back to runtime compilation if needed

### Memory Management
- Use MTLBuffer for tensor storage
- Prefer shared storage mode for unified memory
- Use texture-based convolution for better cache utilization
- Reuse buffers when possible

### Thread Scheduling
- Compute threadgroup size based on operator type
- Use maximum threads per threadgroup (1024 for Apple GPUs)
- Optimize for 2D threadgroups for convolution/pooling
- Use 1D threadgroups for element-wise ops

### Shader Files (to be compiled separately)
- `Conv2D.metal`
- `Relu.metal`
- `MaxPool2D.metal`
- `ElementWise.metal`
- `FullyConnected.metal`
- `BatchNorm.metal`
- `Softmax.metal`

## File Structure
```
src/MobileRuntime/Backends/Metal/
├── MetalBackend.cs
├── MetalBackendFactory.cs
├── MetalBuffer.cs
├── MetalCommandBuffer.cs
├── Interfaces/
│   ├── IMetalBackend.cs
│   ├── IMetalBuffer.cs
│   └── IMetalComputeShader.cs
├── Shaders/
│   ├── MetalConv2DShader.cs
│   ├── MetalReluShader.cs
│   ├── MetalMaxPool2DShader.cs
│   ├── MetalElementWiseShader.cs
│   ├── MetalFullyConnectedShader.cs
│   ├── MetalBatchNormShader.cs
│   └── MetalSoftmaxShader.cs
└── Models/
    ├── MetalBackendCapabilities.cs
    └── MetalDeviceInfo.cs
```

## Success Criteria
- Metal backend initializes successfully on iOS
- All operators execute on GPU
- Results match CPU reference
- No memory leaks
- Performance meets targets

## Dependencies
- spec_mobile_runtime_core (interfaces)
- spec_mobile_tensor_ops (Tensor)
- spec_mobile_model_format (OperatorDescriptor)

## Testing Requirements
- Unit tests for each operator
- Comparison against CPU reference
- Memory leak detection
- Performance benchmarks
- iOS device testing

## Performance Targets
- Conv2D (3x3, 32 channels, 224x224 input): < 3ms
- FullyConnected (1024 -> 1024): < 0.5ms
- MaxPool2D (2x2, 224x224): < 0.5ms
- Element-wise ops (1M elements): < 0.3ms
- GPU > 2x faster than CPU for typical workloads

## Platform Notes
- iOS only (requires Metal.framework)
- Requires iOS 12.0+ (Metal 2.0)
- Requires iOS 11.0+ (Metal 1.0 as fallback)
- Requires physical device (Metal not available in simulator)
