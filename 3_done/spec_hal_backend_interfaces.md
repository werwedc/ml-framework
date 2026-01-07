# Spec: HAL Backend Interfaces

## Overview
Define the backend abstraction layer and registry system.

## Responsibilities
- Create IBackend interface for compute backends
- Implement BackendRegistry for dynamic backend registration
- Define Operation abstraction for supported operations

## Files to Create/Modify
- `src/HAL/IBackend.cs` - Backend interface
- `src/HAL/BackendRegistry.cs` - Backend registry
- `src/HAL/Operation.cs` - Operation abstraction (enum or class)
- `tests/HAL/BackendRegistryTests.cs` - Registry tests

## API Design

### Operation.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// Enumeration of supported tensor operations
/// </summary>
public enum Operation
{
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,

    // Linear Algebra
    MatMul,
    Transpose,

    // Reductions
    Sum,
    Mean,
    Max,
    Min,

    // Memory
    Copy,
    Fill,

    // Activation Functions
    ReLU,
    Sigmoid,
    Tanh,

    // Convolution
    Conv2D,
    MaxPool2D,

    // Misc
    Cast,
    Reshape
}
```

### IBackend.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// Represents a compute backend for a specific device type
/// </summary>
public interface IBackend
{
    /// <summary>
    /// Backend name (e.g., "MKL", "CUDA", "rocBLAS")
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Device type this backend supports
    /// </summary>
    DeviceType Type { get; }

    /// <summary>
    /// Check if this backend supports a specific operation
    /// </summary>
    bool SupportsOperation(Operation operation);

    /// <summary>
    /// Execute an operation on input tensors
    /// </summary>
    /// <param name="operation">Operation to execute</param>
    /// <param name="inputs">Input tensors</param>
    /// <returns>Result tensor</returns>
    Tensor ExecuteOperation(Operation operation, Tensor[] inputs);

    /// <summary>
    /// Initialize the backend (load libraries, etc.)
    /// </summary>
    void Initialize();

    /// <summary>
    /// Check if the backend is available (hardware present, libraries loaded)
    /// </summary>
    bool IsAvailable { get; }
}
```

### BackendRegistry.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// Global registry for compute backends
/// </summary>
public static class BackendRegistry
{
    private static readonly Dictionary<DeviceType, IBackend> _backends = new();
    private static readonly object _registryLock = new();

    /// <summary>
    /// Register a backend for a specific device type
    /// </summary>
    /// <exception cref="ArgumentException">Thrown if backend already registered</exception>
    public static void Register(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);

        lock (_registryLock)
        {
            if (_backends.ContainsKey(backend.Type))
            {
                throw new ArgumentException(
                    $"Backend for device type {backend.Type} already registered");
            }

            _backends[backend.Type] = backend;

            // Initialize the backend
            if (backend.IsAvailable)
            {
                backend.Initialize();
            }
        }
    }

    /// <summary>
    /// Get the registered backend for a device type
    /// </summary>
    /// <returns>Backend instance, or null if not registered</returns>
    public static IBackend? GetBackend(DeviceType type)
    {
        lock (_registryLock)
        {
            return _backends.TryGetValue(type, out var backend) ? backend : null;
        }
    }

    /// <summary>
    /// Get all available device types (with registered backends)
    /// </summary>
    public static IEnumerable<DeviceType> GetAvailableDevices()
    {
        lock (_registryLock)
        {
            return _backends.Values
                .Where(b => b.IsAvailable)
                .Select(b => b.Type)
                .ToList();
        }
    }

    /// <summary>
    /// Check if a device type has a registered and available backend
    /// </summary>
    public static bool IsDeviceAvailable(DeviceType type)
    {
        return GetAvailableDevices().Contains(type);
    }

    /// <summary>
    /// Clear all registered backends (for testing)
    /// </summary>
    public static void Clear()
    {
        lock (_registryLock)
        {
            _backends.Clear();
        }
    }
}
```

## Testing Requirements
```csharp
public class BackendRegistryTests
{
    private class MockBackend : IBackend
    {
        public string Name => "Mock";
        public DeviceType Type { get; }
        public bool IsAvailable => true;

        public MockBackend(DeviceType type)
        {
            Type = type;
        }

        public bool SupportsOperation(Operation operation) => true;
        public Tensor ExecuteOperation(Operation operation, Tensor[] inputs)
            => throw new NotImplementedException();
        public void Initialize() { }
    }

    [Test]
    public void Register_AddsBackend()
    {
        var backend = new MockBackend(DeviceType.CPU);
        BackendRegistry.Register(backend);

        Assert.AreSame(backend, BackendRegistry.GetBackend(DeviceType.CPU));
    }

    [Test]
    public void Register_DuplicateType_ThrowsException()
    {
        var backend1 = new MockBackend(DeviceType.CPU);
        var backend2 = new MockBackend(DeviceType.CPU);

        BackendRegistry.Register(backend1);

        Assert.Throws<ArgumentException>(() =>
        {
            BackendRegistry.Register(backend2);
        });
    }

    [Test]
    public void GetAvailableDevices_ReturnsRegisteredTypes()
    {
        BackendRegistry.Register(new MockBackend(DeviceType.CPU));
        BackendRegistry.Register(new MockBackend(DeviceType.CUDA));

        var devices = BackendRegistry.GetAvailableDevices();

        CollectionAssert.AreEquivalent(
            new[] { DeviceType.CPU, DeviceType.CUDA },
            devices);
    }

    [SetUp]
    public void Setup()
    {
        BackendRegistry.Clear();
    }
}
```

## Acceptance Criteria
- [ ] Operation enum defines common tensor operations
- [ ] IBackend interface defines required methods
- [ ] BackendRegistry supports registration and lookup
- [ ] Thread-safe registration with proper locking
- [ ] All tests pass
- [ ] XML documentation for all public members

## Notes for Coder
- Tensor type is referenced but not yet defined - will be implemented separately
- Initialize() should be called when registering an available backend
- Focus on the registry pattern and proper error handling
- Ensure thread safety with proper locking
