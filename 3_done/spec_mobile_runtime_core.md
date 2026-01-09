# Spec: Mobile Runtime Core Interfaces

## Overview
Define the core interfaces and base classes for the mobile runtime system.

## Requirements
- Minimal interfaces for model loading and inference
- No training or gradient computation APIs
- Platform-agnostic base classes
- Hardware backend abstraction layer

## Classes to Implement

### 1. `IMobileRuntime` Interface
```csharp
public interface IMobileRuntime
{
    // Model loading
    IModel LoadModel(string modelPath);
    IModel LoadModel(byte[] modelBytes);

    // Configuration
    void SetMemoryLimit(long maxBytes);
    void SetHardwareBackend(BackendType backend);
    BackendType CurrentBackend { get; }
    long MemoryLimit { get; }

    // Runtime info
    RuntimeInfo GetRuntimeInfo();
}

public enum BackendType
{
    CPU,
    GPU,
    NPU,
    Auto
}

public class RuntimeInfo
{
    public string Version { get; set; }
    public BackendType SupportedBackends { get; set; }
    public string Platform { get; set; }
    public string DeviceInfo { get; set; }
}
```

### 2. `IModel` Interface
```csharp
public interface IModel : IDisposable
{
    string Name { get; }
    InputInfo[] Inputs { get; }
    OutputInfo[] Outputs { get; }

    // Inference
    ITensor[] Predict(ITensor[] inputs);
    Task<ITensor[]> PredictAsync(ITensor[] inputs);

    // Model metadata
    long MemoryFootprint { get; }
    ModelFormat Format { get; }
}

public enum ModelFormat
{
    MobileBinary,
    Protobuf,
    ONNX  // Future
}

public class InputInfo
{
    public string Name { get; set; }
    public int[] Shape { get; set; }
    public DataType DataType { get; set; }
}

public class OutputInfo
{
    public string Name { get; set; }
    public int[] Shape { get; set; }
    public DataType DataType { get; set; }
}

public enum DataType
{
    Float32,
    Float16,
    Int8,
    Int16,
    Int32
}
```

### 3. `ITensor` Interface
```csharp
public interface ITensor : IDisposable
{
    int[] Shape { get; }
    DataType DataType { get; }
    long Size { get; }
    long ByteCount { get; }

    // Data access (read-only for runtime)
    T GetData<T>(params int[] indices);
    T[] ToArray<T>();
}
```

### 4. `MobileRuntime` Base Class
```csharp
public abstract class MobileRuntime : IMobileRuntime
{
    protected long _memoryLimit;
    protected BackendType _currentBackend;

    public virtual void SetMemoryLimit(long maxBytes)
    {
        if (maxBytes <= 0)
            throw new ArgumentException("Memory limit must be positive");
        _memoryLimit = maxBytes;
    }

    public virtual void SetHardwareBackend(BackendType backend)
    {
        _currentBackend = backend;
    }

    public abstract IModel LoadModel(string modelPath);
    public abstract IModel LoadModel(byte[] modelBytes);
    public abstract RuntimeInfo GetRuntimeInfo();
}
```

### 5. `Model` Base Class
```csharp
public abstract class Model : IModel
{
    public abstract string Name { get; }
    public abstract InputInfo[] Inputs { get; }
    public abstract OutputInfo[] Outputs { get; }
    public abstract long MemoryFootprint { get; }
    public abstract ModelFormat Format { get; }

    public abstract ITensor[] Predict(ITensor[] inputs);
    public abstract Task<ITensor[]> PredictAsync(ITensor[] inputs);

    public abstract void Dispose();
}
```

## File Structure
```
src/MobileRuntime/
├── Interfaces/
│   ├── IMobileRuntime.cs
│   ├── IModel.cs
│   └── ITensor.cs
├── Models/
│   ├── MobileRuntime.cs
│   └── Model.cs
└── Enums/
    ├── BackendType.cs
    ├── ModelFormat.cs
    └── DataType.cs
```

## Success Criteria
- All interfaces compile
- Base classes provide default implementations
- Enums cover all specified types
- Interfaces are minimal and focused
- No training/gradient APIs present

## Dependencies
- .NET Standard 2.0 (for cross-platform compatibility)
- System.Threading.Tasks

## Testing Requirements
- None in this spec (testing covered in separate spec)
