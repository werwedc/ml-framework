# Spec: HAL Device Types and Device Factory

## Overview
Define device types and a factory for creating device instances.

## Responsibilities
- Create DeviceType enum for supported hardware
- Implement Device factory class for creating device instances
- Support static factory methods for common device configurations

## Files to Create/Modify
- `src/HAL/DeviceType.cs` - Device type enumeration
- `src/HAL/Device.cs` - Device factory class
- `tests/HAL/DeviceFactoryTests.cs` - Factory tests

## API Design

### DeviceType.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// Enumeration of supported device types
/// </summary>
public enum DeviceType
{
    /// <summary>
    /// CPU device (default fallback)
    /// </summary>
    CPU,

    /// <summary>
    /// NVIDIA CUDA GPU
    /// </summary>
    CUDA,

    /// <summary>
    /// AMD ROCm GPU
    /// </summary>
    ROCm,

    /// <summary>
    /// Apple Metal GPU
    /// </summary>
    Metal,

    /// <summary>
    /// Intel oneAPI GPU
    /// </summary>
    OneAPI
}
```

### Device.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// Factory class for creating device instances
/// </summary>
public static class Device
{
    private static readonly Dictionary<(DeviceType type, int id), IDevice> _deviceCache = new();
    private static readonly object _cacheLock = new();

    /// <summary>
    /// Get the default CPU device (device ID 0)
    /// </summary>
    public static IDevice CPU => GetDevice(DeviceType.CPU, 0);

    /// <summary>
    /// Get a specific CUDA device
    /// </summary>
    /// <param name="deviceId">Device ID (0, 1, 2, ...)</param>
    public static IDevice CUDA(int deviceId = 0)
    {
        return GetDevice(DeviceType.CUDA, deviceId);
    }

    /// <summary>
    /// Get a specific ROCm device
    /// </summary>
    /// <param name="deviceId">Device ID (0, 1, 2, ...)</param>
    public static IDevice ROCm(int deviceId = 0)
    {
        return GetDevice(DeviceType.ROCm, deviceId);
    }

    /// <summary>
    /// Get a specific Metal device
    /// </summary>
    /// <param name="deviceId">Device ID (0, 1, 2, ...)</param>
    public static IDevice Metal(int deviceId = 0)
    {
        return GetDevice(DeviceType.Metal, deviceId);
    }

    /// <summary>
    /// Get a specific oneAPI device
    /// </summary>
    /// <param name="deviceId">Device ID (0, 1, 2, ...)</param>
    public static IDevice OneAPI(int deviceId = 0)
    {
        return GetDevice(DeviceType.OneAPI, deviceId);
    }

    /// <summary>
    /// Get a device instance by type and ID
    /// Devices are cached - repeated calls return the same instance
    /// </summary>
    public static IDevice GetDevice(DeviceType type, int deviceId = 0)
    {
        var key = (type, deviceId);

        lock (_cacheLock)
        {
            if (!_deviceCache.TryGetValue(key, out var device))
            {
                device = CreateDevice(type, deviceId);
                _deviceCache[key] = device;
            }
            return device;
        }
    }

    /// <summary>
    /// Clear the device cache (for testing purposes)
    /// </summary>
    public static void ClearCache()
    {
        lock (_cacheLock)
        {
            _deviceCache.Clear();
        }
    }

    private static IDevice CreateDevice(DeviceType type, int deviceId)
    {
        // This will be implemented when we create concrete device classes
        // For now, throw NotImplementedException
        throw new NotImplementedException($"Device type {type} not yet implemented");
    }
}
```

## Testing Requirements
```csharp
public class DeviceFactoryTests
{
    [Test]
    public void CPU_ReturnsSameInstanceOnMultipleCalls()
    {
        var device1 = Device.CPU;
        var device2 = Device.CPU;

        Assert.AreSame(device1, device2);
    }

    [Test]
    public void GetDevice_CachesDevices()
    {
        var device1 = Device.GetDevice(DeviceType.CPU, 0);
        var device2 = Device.GetDevice(DeviceType.CPU, 0);

        Assert.AreSame(device1, device2);
    }

    [Test]
    public void GetDevice_DifferentIds_DifferentInstances()
    {
        var device1 = Device.GetDevice(DeviceType.CPU, 0);
        var device2 = Device.GetDevice(DeviceType.CPU, 1);

        Assert.AreNotSame(device1, device2);
    }

    [Test]
    public void ClearCache_ResetsCache()
    {
        var device1 = Device.CPU;
        Device.ClearCache();
        var device2 = Device.CPU;

        Assert.AreNotSame(device1, device2);
    }
}
```

## Acceptance Criteria
- [ ] DeviceType enum defines all required types
- [ ] Device factory provides static methods for common devices (CPU, CUDA, etc.)
- [ ] Device caching works correctly (same instance returned for same type/id)
- [ ] ClearCache() method available for testing
- [ ] All tests pass
- [ ] XML documentation for all public members

## Notes for Coder
- Note: CreateDevice() currently throws NotImplementedException
- This is expected - will be implemented when CPU/GPU backends are created
- Focus on the factory pattern and caching logic
- Thread-safe caching using lock is required
