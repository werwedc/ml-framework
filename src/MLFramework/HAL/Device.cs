using MLFramework.HAL.CUDA;

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
        return type switch
        {
            DeviceType.CPU => new CpuDevice(deviceId),
            DeviceType.CUDA => new CudaDevice(deviceId),
            _ => throw new NotImplementedException($"Device type {type} not yet implemented")
        };
    }
}
