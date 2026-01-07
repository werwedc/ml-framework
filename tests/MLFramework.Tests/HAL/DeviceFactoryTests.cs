using Xunit;

namespace MLFramework.HAL.Tests;

public class DeviceFactoryTests
{
    [Fact]
    public void CPU_ReturnsSameInstanceOnMultipleCalls()
    {
        var device1 = Device.CPU;
        var device2 = Device.CPU;

        Assert.Same(device1, device2);
    }

    [Fact]
    public void GetDevice_CachesDevices()
    {
        var device1 = Device.GetDevice(DeviceType.CPU, 0);
        var device2 = Device.GetDevice(DeviceType.CPU, 0);

        Assert.Same(device1, device2);
    }

    [Fact]
    public void GetDevice_DifferentIds_DifferentInstances()
    {
        var device1 = Device.GetDevice(DeviceType.CPU, 0);
        var device2 = Device.GetDevice(DeviceType.CPU, 1);

        Assert.NotSame(device1, device2);
    }

    [Fact]
    public void ClearCache_ResetsCache()
    {
        var device1 = Device.CPU;
        Device.ClearCache();
        var device2 = Device.CPU;

        Assert.NotSame(device1, device2);
    }
}
