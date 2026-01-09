# Spec: Vulkan GPU Backend (Android)

## Overview
Implement Vulkan-based GPU backend for Android devices with cross-platform hardware acceleration.

## Requirements
- Vulkan 1.1+ support
- Compute pipelines for all operators
- SPIR-V shader compilation
- Buffer-based memory management
- Descriptor set optimization
- Cross-platform Android support

## Classes to Implement

### 1. `IVulkanBackend` Interface
```csharp
public interface IVulkanBackend
{
    string Name { get; }
    VulkanBackendCapabilities Capabilities { get; }

    // Memory management
    IVulkanBuffer AllocateBuffer(long size, BufferUsage usage);
    void FreeBuffer(IVulkanBuffer buffer);

    // Operator execution
    ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters);

    // Batch execution
    ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry);

    // Compute pipelines
    IVulkanComputePipeline GetComputePipeline(OperatorType opType);

    // Synchronization
    void WaitForCompletion();

    // Platform info
    VulkanDeviceInfo GetDeviceInfo();
}

public class VulkanBackendCapabilities
{
    public bool SupportsComputeShaders { get; set; }
    public bool SupportsPushDescriptors { get; set; }
    public int MaxComputeWorkGroupInvocations { get; set; }
    public int MaxComputeSharedMemorySize { get; set; }
    public int MaxUniformBufferRange { get; set; }
    public int MaxStorageBufferRange { get; set; }
}

public class VulkanDeviceInfo
{
    public string DeviceName { get; set; }
    public uint VendorId { get; set; }
    public uint DeviceId { get; set; }
    public string DriverVersion { get; set; }
    public VulkanBackendCapabilities Capabilities { get; set; }
}
```

### 2. `VulkanBackend` Class
```csharp
public sealed class VulkanBackend : IVulkanBackend, IDisposable
{
    private IntPtr _instance;
    private IntPtr _physicalDevice;
    private IntPtr _device;
    private IntPtr _commandPool;
    private IntPtr _queue;
    private readonly Dictionary<OperatorType, IVulkanComputePipeline> _pipelines;
    private readonly List<IVulkanBuffer> _allocatedBuffers;
    private readonly ITensorFactory _tensorFactory;

    public VulkanBackend(ITensorFactory tensorFactory);
    ~VulkanBackend();

    public string Name => "Vulkan";
    public VulkanBackendCapabilities Capabilities { get; private set; }

    public IVulkanBuffer AllocateBuffer(long size, BufferUsage usage);
    public void FreeBuffer(IVulkanBuffer buffer);

    public ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters);
    public ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry);

    public IVulkanComputePipeline GetComputePipeline(OperatorType opType);
    public void WaitForCompletion();

    public VulkanDeviceInfo GetDeviceInfo();

    public void Dispose();
    private void Dispose(bool disposing);

    private void InitializeVulkan();
    private void InitializePipelines();
    private void SelectPhysicalDevice();
    private void DetectCapabilities();
}
```

### 3. `IVulkanBuffer` Interface
```csharp
public interface IVulkanBuffer : IDisposable
{
    IntPtr NativeBuffer { get; }
    IntPtr DeviceMemory { get; }
    long Length { get; }
    BufferUsage Usage { get; }

    void CopyFrom(IntPtr source, long size);
    void CopyTo(IntPtr destination, long size);
    void Map();
    void Unmap();
}

public enum BufferUsage
{
    Storage,
    Uniform,
    Staging
}
```

### 4. `VulkanBuffer` Class
```csharp
public sealed class VulkanBuffer : IVulkanBuffer
{
    private readonly IntPtr _device;
    private IntPtr _buffer;
    private IntPtr _deviceMemory;
    private readonly long _length;
    private readonly BufferUsage _usage;
    private IntPtr _mappedMemory;

    public VulkanBuffer(IntPtr device, long length, BufferUsage usage);
    ~VulkanBuffer();

    public IntPtr NativeBuffer => _buffer;
    public IntPtr DeviceMemory => _deviceMemory;
    public long Length => _length;
    public BufferUsage Usage => _usage;

    public void CopyFrom(IntPtr source, long size);
    public void CopyTo(IntPtr destination, long size);
    public void Map();
    public void Unmap();

    public void Dispose();
    private void Dispose(bool disposing);

    private void AllocateBuffer();
    private void AllocateDeviceMemory();
    private void BindDeviceMemory();
}
```

### 5. `IVulkanComputePipeline` Interface
```csharp
public interface IVulkanComputePipeline : IDisposable
{
    string Name { get; }
    OperatorType OperatorType { get; }

    void Dispatch(VulkanCommandBuffer commandBuffer, VulkanBuffer[] inputs, VulkanBuffer[] outputs, Dictionary<string, object> parameters);
    void UpdateDescriptorSets(VulkanBuffer[] buffers, Dictionary<string, object> parameters);
}
```

### 6. Compute Pipelines

#### VulkanConv2DPipeline
```csharp
public sealed class VulkanConv2DPipeline : IVulkanComputePipeline
{
    private readonly VulkanBackend _backend;
    private readonly IntPtr _pipeline;
    private readonly IntPtr _pipelineLayout;
    private readonly IntPtr _descriptorSetLayout;

    public string Name => "Conv2D";
    public OperatorType OperatorType => OperatorType.Conv2D;

    public VulkanConv2DPipeline(VulkanBackend backend);

    public void Dispatch(VulkanCommandBuffer commandBuffer, VulkanBuffer[] inputs, VulkanBuffer[] outputs, Dictionary<string, object> parameters);
    public void UpdateDescriptorSets(VulkanBuffer[] buffers, Dictionary<string, object> parameters);

    public void Dispose();

    private void CreatePipeline();
    private void CreateDescriptorSetLayout();
    private void CreatePipelineLayout();
}
```

#### VulkanReluPipeline
```csharp
public sealed class VulkanReluPipeline : IVulkanComputePipeline
{
    private readonly VulkanBackend _backend;
    private readonly IntPtr _pipeline;
    private readonly IntPtr _pipelineLayout;
    private readonly IntPtr _descriptorSetLayout;

    public string Name => "Relu";
    public OperatorType OperatorType => OperatorType.Relu;

    public VulkanReluPipeline(VulkanBackend backend);

    public void Dispatch(VulkanCommandBuffer commandBuffer, VulkanBuffer[] inputs, VulkanBuffer[] outputs, Dictionary<string, object> parameters);
    public void UpdateDescriptorSets(VulkanBuffer[] buffers, Dictionary<string, object> parameters);

    public void Dispose();

    private void CreatePipeline();
    private void CreateDescriptorSetLayout();
    private void CreatePipelineLayout();
}
```

#### VulkanMaxPool2DPipeline
```csharp
public sealed class VulkanMaxPool2DPipeline : IVulkanComputePipeline
{
    private readonly VulkanBackend _backend;
    private readonly IntPtr _pipeline;
    private readonly IntPtr _pipelineLayout;
    private readonly IntPtr _descriptorSetLayout;

    public string Name => "MaxPool2D";
    public OperatorType OperatorType => OperatorType.MaxPool2D;

    public VulkanMaxPool2DPipeline(VulkanBackend backend);

    public void Dispatch(VulkanCommandBuffer commandBuffer, VulkanBuffer[] inputs, VulkanBuffer[] outputs, Dictionary<string, object> parameters);
    public void UpdateDescriptorSets(VulkanBuffer[] buffers, Dictionary<string, object> parameters);

    public void Dispose();

    private void CreatePipeline();
    private void CreateDescriptorSetLayout();
    private void CreatePipelineLayout();
}
```

#### VulkanElementWisePipeline
```csharp
public sealed class VulkanElementWisePipeline : IVulkanComputePipeline
{
    private readonly VulkanBackend _backend;
    private readonly IntPtr _pipeline;
    private readonly IntPtr _pipelineLayout;
    private readonly IntPtr _descriptorSetLayout;
    private readonly OperatorType _opType;

    public string Name => _opType.ToString();
    public OperatorType OperatorType => _opType;

    public VulkanElementWisePipeline(VulkanBackend backend, OperatorType opType);

    public void Dispatch(VulkanCommandBuffer commandBuffer, VulkanBuffer[] inputs, VulkanBuffer[] outputs, Dictionary<string, object> parameters);
    public void UpdateDescriptorSets(VulkanBuffer[] buffers, Dictionary<string, object> parameters);

    public void Dispose();

    private void CreatePipeline();
    private void CreateDescriptorSetLayout();
    private void CreatePipelineLayout();
}
```

### 7. `VulkanCommandBuffer` Helper Class
```csharp
public sealed class VulkanCommandBuffer : IDisposable
{
    private readonly IntPtr _commandBuffer;
    private readonly VulkanBackend _backend;
    private bool _recording;

    public VulkanCommandBuffer(VulkanBackend backend);

    public void Begin();
    public void End();
    public void Submit();
    public void WaitUntilCompleted();

    public void BindPipeline(IntPtr pipeline);
    public void BindDescriptorSets(IntPtr pipelineLayout, IntPtr descriptorSet);
    public void Dispatch(uint groupCountX, uint groupCountY, uint groupCountZ);
    public void CopyBuffer(IntPtr srcBuffer, IntPtr dstBuffer, long size);
    public void BufferMemoryBarrier(IntPtr buffer, VkAccessFlags srcAccess, VkAccessFlags dstAccess);

    public void Dispose();
}

[Flags]
public enum VkAccessFlags
{
    ShaderRead = 1 << 0,
    ShaderWrite = 1 << 1,
    TransferRead = 1 << 2,
    TransferWrite = 1 << 3,
    HostRead = 1 << 4,
    HostWrite = 1 << 5
}
```

### 8. `VulkanBackendFactory` Class
```csharp
public static class VulkanBackendFactory
{
    public static IVulkanBackend CreateDefault(ITensorFactory tensorFactory);
    public static IVulkanBackend CreateForDevice(uint vendorId, uint deviceId, ITensorFactory tensorFactory);
    public static bool IsAvailable();
    public static VulkanDeviceInfo[] EnumerateDevices();
}
```

## Implementation Notes

### Vulkan Interop
- Use P/Invoke to call Vulkan APIs
- Link against libvulkan.so on Android
- Use `[DllImport("vulkan")]` for calls
- Handle VkResult errors appropriately

### Shader Compilation
- Use GLSL compute shaders
- Compile GLSL to SPIR-V offline using glslc
- Load pre-compiled SPIR-V binaries at runtime
- Use shader specialization constants for parameters

### Memory Management
- Use VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT for GPU-only buffers
- Use VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT for staging buffers
- Use VK_MEMORY_PROPERTY_HOST_COHERENT_BIT for zero-copy on unified memory
- Use memory heaps efficiently

### Descriptor Sets
- Pre-create descriptor set layouts for each operator
- Use push descriptors for small parameter sets (if supported)
- Reuse descriptor sets when possible
- Use dynamic uniform buffers for variable sizes

### Thread Scheduling
- Compute workgroup size based on operator and GPU limits
- Use maximum workgroup invocations (usually 1024)
- Optimize for 2D/3D workgroups for convolution/pooling
- Use 1D workgroups for element-wise ops

### Shader Files (GLSL to be compiled to SPIR-V)
- `conv2d.comp`
- `relu.comp`
- `maxpool2d.comp`
- `elementwise.comp`
- `fullyconnected.comp`
- `batchnorm.comp`
- `softmax.comp`

## File Structure
```
src/MobileRuntime/Backends/Vulkan/
├── VulkanBackend.cs
├── VulkanBackendFactory.cs
├── VulkanBuffer.cs
├── VulkanCommandBuffer.cs
├── Interfaces/
│   ├── IVulkanBackend.cs
│   ├── IVulkanBuffer.cs
│   └── IVulkanComputePipeline.cs
├── Pipelines/
│   ├── VulkanConv2DPipeline.cs
│   ├── VulkanReluPipeline.cs
│   ├── VulkanMaxPool2DPipeline.cs
│   ├── VulkanElementWisePipeline.cs
│   ├── VulkanFullyConnectedPipeline.cs
│   ├── VulkanBatchNormPipeline.cs
│   └── VulkanSoftmaxPipeline.cs
└── Models/
    ├── VulkanBackendCapabilities.cs
    └── VulkanDeviceInfo.cs
```

## Success Criteria
- Vulkan backend initializes successfully on Android
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
- Android device testing (multiple GPUs)

## Performance Targets
- Conv2D (3x3, 32 channels, 224x224 input): < 3ms
- FullyConnected (1024 -> 1024): < 0.5ms
- MaxPool2D (2x2, 224x224): < 0.5ms
- Element-wise ops (1M elements): < 0.3ms
- GPU > 2x faster than CPU for typical workloads

## Platform Notes
- Android only (requires libvulkan.so)
- Requires Android 7.0+ (Vulkan 1.0) or Android 9.0+ (Vulkan 1.1)
- Requires Vulkan-capable GPU
- Falls back to CPU backend if Vulkan unavailable
