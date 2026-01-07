# Spec: Device Mesh Management

## Overview
Implement the device mesh abstraction for parallel execution. This provides the foundation for distributing computations across multiple devices.

## Scope
- Define Device and DeviceMesh classes
- Implement mesh topology (1D, 2D, etc.)
- Support device selection and allocation
- Define axis-based sharding

## Technical Requirements

### 1. Device Class

```csharp
namespace MLFramework.Functional.Distributed
{
    /// <summary>
    /// Represents a computation device (CPU, GPU, TPU, etc.).
    /// </summary>
    public class Device
    {
        public int Id { get; }
        public DeviceType Type { get; }
        public string Name { get; }
        public bool IsAvailable { get; }

        public Device(int id, DeviceType type, string name, bool isAvailable = true)
        {
            Id = id;
            Type = type;
            Name = name;
            IsAvailable = isAvailable;
        }

        public override string ToString() => $"Device({Type}:{Id})";

        // Factory methods for common device types
        public static Device CPU(int id) => new Device(id, DeviceType.CPU, $"cpu:{id}");
        public static Device GPU(int id) => new Device(id, DeviceType.GPU, $"gpu:{id}");
    }

    public enum DeviceType
    {
        CPU,
        GPU,
        TPU,
        Other
    }
}
```

### 2. DeviceMesh Class

```csharp
/// <summary>
/// A mesh of devices for distributed computation.
/// Supports N-dimensional topologies for data and model parallelism.
/// </summary>
public class DeviceMesh
{
    private readonly Device[] _devices;
    private readonly int[] _shape;
    private readonly Dictionary<string, Device> _deviceMap;

    /// <summary>
    /// Shape of the mesh (e.g., [8] for 1D, [2, 4] for 2D).
    /// </summary>
    public IReadOnlyList<int> Shape => _shape;

    /// <summary>
    /// Total number of devices in the mesh.
    /// </summary>
    public int DeviceCount => _devices.Length;

    /// <summary>
    /// Number of dimensions in the mesh.
    /// </summary>
    public int Rank => _shape.Length;

    /// <summary>
    /// Create a 1D device mesh.
    /// </summary>
    public DeviceMesh(Device[] devices)
    {
        if (devices == null || devices.Length == 0)
            throw new ArgumentException("Must provide at least one device");

        _devices = devices.ToArray();
        _shape = new int[] { _devices.Length };
        _deviceMap = new Dictionary<string, Device>();

        for (int i = 0; i < _devices.Length; i++)
        {
            _deviceMap[_devices[i].Name] = _devices[i];
        }
    }

    /// <summary>
    /// Create an N-dimensional device mesh.
    /// </summary>
    /// <param name="shape">Shape of the mesh (e.g., [2, 4] for 2D mesh with 8 devices)</param>
    /// <param name="devices">List of devices to use</param>
    public DeviceMesh(int[] shape, Device[] devices)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape must have at least one dimension");

        int totalDevices = shape.Aggregate(1, (a, b) => a * b);
        if (devices.Length != totalDevices)
        {
            throw new ArgumentException($"Device count ({devices.Length}) must match mesh size ({totalDevices})");
        }

        _shape = shape.ToArray();
        _devices = devices.ToArray();
        _deviceMap = new Dictionary<string, Device>();

        for (int i = 0; i < _devices.Length; i++)
        {
            _deviceMap[_devices[i].Name] = _devices[i];
        }
    }

    /// <summary>
    /// Get device at specific mesh coordinates.
    /// </summary>
    public Device GetDevice(params int[] indices)
    {
        if (indices.Length != Rank)
            throw new ArgumentException($"Expected {Rank} indices, got {indices.Length}");

        int flatIndex = ComputeFlatIndex(indices);
        return _devices[flatIndex];
    }

    /// <summary>
    /// Get all devices along a specific axis.
    /// </summary>
    public Device[] GetAxisDevices(int axis)
    {
        if (axis < 0 || axis >= Rank)
            throw new ArgumentException($"Invalid axis {axis} for mesh of rank {Rank}");

        return _devices.Distinct().ToArray();  // Simplified - in reality, group by axis
    }

    /// <summary>
    /// Sharding axis for distributing data.
    /// </summary>
    public class ShardingAxis
    {
        public string Name { get; }
        public int AxisIndex { get; }

        public ShardingAxis(string name, int axisIndex)
        {
            Name = name;
            AxisIndex = axisIndex;
        }
    }

    /// <summary>
    /// Get a sharding axis by name.
    /// </summary>
    public ShardingAxis this[string name]
    {
        get
        {
            if (name == "data" || name == "batch")
            {
                return new ShardingAxis(name, 0);  // Default to first axis
            }
            throw new ArgumentException($"Unknown axis: {name}");
        }
    }

    private int ComputeFlatIndex(int[] indices)
    {
        // Convert multi-dimensional index to flat array index (row-major order)
        int index = 0;
        int stride = 1;

        for (int i = Rank - 1; i >= 0; i--)
        {
            if (indices[i] < 0 || indices[i] >= _shape[i])
                throw new IndexOutOfRangeException($"Index {indices[i]} out of range for axis {i}");

            index += indices[i] * stride;
            stride *= _shape[i];
        }

        return index;
    }
}
```

### 3. DeviceMesh Factory

```csharp
public static class DeviceMeshFactory
{
    /// <summary>
    /// Create a 1D mesh of devices.
    /// </summary>
    public static DeviceMesh Create1D(int deviceCount, DeviceType type = DeviceType.CPU)
    {
        var devices = new Device[deviceCount];
        for (int i = 0; i < deviceCount; i++)
        {
            devices[i] = type == DeviceType.CPU ? Device.CPU(i) : Device.GPU(i);
        }
        return new DeviceMesh(devices);
    }

    /// <summary>
    /// Create a 2D mesh of devices (e.g., for data parallelism and model parallelism).
    /// </summary>
    public static DeviceMesh Create2D(int rows, int cols, DeviceType type = DeviceType.CPU)
    {
        int totalDevices = rows * cols;
        var devices = new Device[totalDevices];
        for (int i = 0; i < totalDevices; i++)
        {
            devices[i] = type == DeviceType.CPU ? Device.CPU(i) : Device.GPU(i);
        }
        return new DeviceMesh(new int[] { rows, cols }, devices);
    }

    /// <summary>
    /// Get default mesh for the system.
    /// </summary>
    public static DeviceMesh Default()
    {
        // In reality, this would query available devices
        // For now, return a single CPU mesh
        return Create1D(1, DeviceType.CPU);
    }
}
```

### 4. Mesh Axis Naming

```csharp
public static class MeshAxisNames
{
    /// <summary>
    /// Data/batch parallelism axis.
    /// </summary>
    public const string Data = "data";

    /// <summary>
    /// Model parallelism axis (e.g., for tensor parallelism).
    /// </summary>
    public const string Model = "model";

    /// <summary>
    /// Pipeline parallelism axis.
    /// </summary>
    public const string Pipeline = "pipeline";
}
```

## Files to Create
1. `src/MLFramework/Functional/Distributed/Device.cs`
2. `src/MLFramework/Functional/Distributed/DeviceMesh.cs`
3. `src/MLFramework/Functional/Distributed/DeviceMeshFactory.cs`
4. `src/MLFramework/Functional/Distributed/MeshAxisNames.cs`

## Dependencies
- spec_functional_core_interfaces.md (for TransformationType enum)

## Success Criteria
- Can create 1D and 2D device meshes
- Can retrieve devices by index
- Can map named axes to mesh dimensions
- Proper validation for mesh shapes and device counts

## Notes for Coder
- This is infrastructure - keep it simple for now
- Device availability checking can be a placeholder for now
- Mesh communication primitives will be added in the next spec
- Focus on the abstraction, not actual device management yet
- Use simple linear indexing for 1D meshes

## Example Usage
```csharp
// 1D mesh of 8 devices
var mesh1D = DeviceMeshFactory.Create1D(8, DeviceType.GPU);

// 2D mesh for data + model parallelism
var mesh2D = DeviceMeshFactory.Create2D(2, 4, DeviceType.GPU);  // 2x4 grid

// Access device
var device = mesh2D.GetDevice(0, 1);

// Get sharding axis
var dataAxis = mesh2D["data"];
```
