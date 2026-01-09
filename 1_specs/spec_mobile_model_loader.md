# Spec: Mobile Model Loader and Execution

## Overview
Implement model loading and inference execution for the mobile runtime, integrating all backends and components.

## Requirements
- Load models from files and byte arrays
- Execute inference on configured backend
- Support synchronous and async inference
- Memory-aware model loading
- Error handling and validation
- Performance metrics collection

## Classes to Implement

### 1. `MobileModel` Class
```csharp
public sealed class MobileModel : Model
{
    private readonly IBackend _backend;
    private readonly MobileModelFormat _modelFormat;
    private readonly Dictionary<uint, ITensor> _constantTensors;
    private readonly Dictionary<uint, ITensor> _runtimeTensors;
    private readonly OperatorDescriptor[] _operators;
    private readonly MemoryPoolStats _memoryStats;
    private readonly InferenceMetrics _metrics;

    public MobileModel(string filePath, IBackend backend);
    public MobileModel(byte[] modelData, IBackend backend);
    ~MobileModel();

    public override string Name => _modelFormat.Metadata.Name;
    public override InputInfo[] Inputs => ToInputInfo(_modelFormat.Inputs);
    public override OutputInfo[] Outputs => ToOutputInfo(_modelFormat.Outputs);
    public override long MemoryFootprint => _memoryStats.PeakUsage;
    public override ModelFormat Format => ModelFormat.MobileBinary;

    public override ITensor[] Predict(ITensor[] inputs);
    public override Task<ITensor[]> PredictAsync(ITensor[] inputs);

    public MemoryPoolStats GetMemoryStats() => _memoryStats;
    public InferenceMetrics GetMetrics() => _metrics;
    public void ResetMetrics();

    public override void Dispose();

    private void LoadModel(MobileModelFormat modelFormat);
    private ITensor[] ExecuteInference(ITensor[] inputs);
    private void ValidateInputs(ITensor[] inputs);
    private ITensor[] PrepareInputs(ITensor[] inputs);
    private ITensor[] ExtractOutputs();
    private void UpdateMetrics(TimeSpan elapsed);

    private static InputInfo ToInputInfo(InputOutputSpec spec);
    private static OutputInfo ToOutputInfo(InputOutputSpec spec);
}
```

### 2. `InferenceMetrics` Class
```csharp
public class InferenceMetrics
{
    public long TotalInferences { get; set; }
    public TimeSpan TotalInferenceTime { get; set; }
    public TimeSpan AverageInferenceTime { get; set; }
    public TimeSpan MinInferenceTime { get; set; }
    public TimeSpan MaxInferenceTime { get; set; }
    public long TotalInputBytes { get; set; }
    public long TotalOutputBytes { get; set; }
    public DateTime FirstInferenceTime { get; set; }
    public DateTime LastInferenceTime { get; set; }

    public void RecordInference(TimeSpan elapsed, long inputBytes, long outputBytes);
    public void Reset();
}
```

### 3. `RuntimeMobileRuntime` Class
```csharp
public sealed class RuntimeMobileRuntime : MobileRuntime
{
    private readonly ITensorFactory _tensorFactory;
    private readonly Dictionary<BackendType, IBackend> _backends;
    private IBackend _currentBackend;

    public RuntimeMobileRuntime(ITensorFactory tensorFactory = null);

    public override IModel LoadModel(string modelPath);
    public override IModel LoadModel(byte[] modelBytes);
    public override RuntimeInfo GetRuntimeInfo();

    protected override IBackend CreateBackend(BackendType backendType);
    protected override void InitializeBackends();

    private ITensorFactory CreateTensorFactory();
}
```

### 4. `IBackend` Interface (unified)
```csharp
public interface IBackend
{
    BackendType Type { get; }
    string Name { get; }

    ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters);
    ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry);

    bool IsAvailable();
    void Dispose();
}
```

### 5. `BackendAdapter` Classes

#### CpuBackendAdapter
```csharp
public sealed class CpuBackendAdapter : IBackend
{
    private readonly ICpuBackend _cpuBackend;

    public CpuBackendAdapter(ICpuBackend cpuBackend);

    public BackendType Type => BackendType.CPU;
    public string Name => _cpuBackend.Name;

    public ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters);
    public ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry);

    public bool IsAvailable() => true;
    public void Dispose();
}
```

#### MetalBackendAdapter
```csharp
public sealed class MetalBackendAdapter : IBackend
{
    private readonly IMetalBackend _metalBackend;

    public MetalBackendAdapter(IMetalBackend metalBackend);

    public BackendType Type => BackendType.GPU;
    public string Name => _metalBackend.Name;

    public ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters);
    public ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry);

    public bool IsAvailable() => MetalBackendFactory.IsAvailable();
    public void Dispose();
}
```

#### VulkanBackendAdapter
```csharp
public sealed class VulkanBackendAdapter : IBackend
{
    private readonly IVulkanBackend _vulkanBackend;

    public VulkanBackendAdapter(IVulkanBackend vulkanBackend);

    public BackendType Type => BackendType.GPU;
    public string Name => _vulkanBackend.Name;

    public ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters);
    public ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry);

    public bool IsAvailable() => VulkanBackendFactory.IsAvailable();
    public void Dispose();
}
```

### 6. `InferenceExecutor` Class
```csharp
internal sealed class InferenceExecutor
{
    private readonly IBackend _backend;
    private readonly Dictionary<uint, ITensor> _tensorRegistry;

    public InferenceExecutor(IBackend backend);

    public ITensor[] Execute(OperatorDescriptor[] operators, ITensor[] inputs);

    private void RegisterTensors(ITensor[] inputs, InputOutputSpec[] inputSpecs);
    private ITensor[] ExecuteOperators(OperatorDescriptor[] operators);
    private ITensor[] GetOutputs(OutputInfo[] outputSpecs);
}
```

### 7. `ModelValidator` Class
```csharp
public static class ModelValidator
{
    public static ValidationResult ValidateModel(MobileModelFormat modelFormat);

    private static bool ValidateHeader(ModelHeader header);
    private static bool ValidateInputOutputSpecs(InputOutputSpec[] specs);
    private static bool ValidateOperators(OperatorDescriptor[] operators, ConstantTensor[] constantTensors);
    private static bool ValidateTensorIds(OperatorDescriptor[] operators, ConstantTensor[] constantTensors);
}

public class ValidationResult
{
    public bool IsValid { get; set; }
    public List<string> Errors { get; set; }
    public List<string> Warnings { get; set; }
}
```

## Implementation Notes

### Model Loading Process
1. Parse model format
2. Validate model structure
3. Load constant tensors into memory pool
4. Initialize operator execution graph
5. Prepare runtime tensor registry
6. Record memory statistics

### Inference Process
1. Validate input shapes and types
2. Copy inputs to GPU (if using GPU backend)
3. Execute operators in order
4. Copy outputs from GPU (if using GPU backend)
5. Return output tensors
6. Update metrics

### Error Handling
- Throw `ModelLoadException` on load errors
- Throw `InferenceException` on inference errors
- Provide detailed error messages
- Log warnings for non-critical issues

### Memory Management
- Use tensor pool for intermediate tensors
- Reuse buffers across inferences
- Release memory on Dispose
- Track peak usage for reporting

### Performance Optimization
- Pre-allocate input/output buffers
- Reuse command buffers (GPU)
- Batch operator execution when possible
- Use async I/O for file loading

## File Structure
```
src/MobileRuntime/Models/
├── MobileModel.cs
├── RuntimeMobileRuntime.cs
├── InferenceExecutor.cs
├── ModelValidator.cs
└── Metrics/
    ├── InferenceMetrics.cs
    └── Exceptions/
        ├── ModelLoadException.cs
        └── InferenceException.cs
src/MobileRuntime/Backends/Adapters/
├── CpuBackendAdapter.cs
├── MetalBackendAdapter.cs
└── VulkanBackendAdapter.cs
```

## Success Criteria
- Models load successfully from files and bytes
- Inference executes correctly on all backends
- Results match expected outputs
- Memory usage stays within limits
- Performance metrics are accurate
- Error handling is robust

## Dependencies
- spec_mobile_runtime_core (interfaces)
- spec_mobile_tensor_ops (Tensor, TensorOperations)
- spec_mobile_memory_pool (IMemoryPool)
- spec_mobile_model_format (MobileModelFormat)
- spec_mobile_backend_cpu (ICpuBackend)
- spec_mobile_backend_metal (IMetalBackend)
- spec_mobile_backend_vulkan (IVulkanBackend)

## Testing Requirements
- Unit tests for model loading
- Unit tests for inference execution
- Integration tests with all backends
- Error handling tests
- Memory usage tests
- Performance benchmarks
- Cross-platform tests

## Performance Targets
- Model load time: < 100ms for MNIST, < 200ms for CIFAR
- Inference latency: < 20ms for typical models
- Memory footprint: < 50MB for MNIST/CIFAR models
- Async inference overhead: < 1ms
