using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using Xunit;

namespace MLFramework.HAL.Tests;

public class BackendRegistryTests
{
    private class MockBackend : IBackend
    {
        public string Name => "Mock";
        public DeviceType Type { get; }
        public bool IsAvailable { get; set; }

        public MockBackend(DeviceType type, bool isAvailable = true)
        {
            Type = type;
            IsAvailable = isAvailable;
        }

        public bool SupportsOperation(Operation operation) => true;

        public Tensor ExecuteOperation(Operation operation, Tensor[] inputs)
            => throw new NotImplementedException();

        public void Initialize() { }
    }

    [Fact]
    public void Register_AddsBackend()
    {
        var backend = new MockBackend(DeviceType.CPU);
        BackendRegistry.Register(backend);

        Assert.Same(backend, BackendRegistry.GetBackend(DeviceType.CPU));
    }

    [Fact]
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

    [Fact]
    public void Register_NullBackend_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
        {
            BackendRegistry.Register(null!);
        });
    }

    [Fact]
    public void GetAvailableDevices_ReturnsRegisteredTypes()
    {
        BackendRegistry.Register(new MockBackend(DeviceType.CPU));
        BackendRegistry.Register(new MockBackend(DeviceType.CUDA));

        var devices = BackendRegistry.GetAvailableDevices();

        Assert.Contains(DeviceType.CPU, devices);
        Assert.Contains(DeviceType.CUDA, devices);
        Assert.Equal(2, devices.Count());
    }

    [Fact]
    public void GetAvailableDevices_OnlyReturnsAvailableBackends()
    {
        BackendRegistry.Register(new MockBackend(DeviceType.CPU, isAvailable: true));
        BackendRegistry.Register(new MockBackend(DeviceType.CUDA, isAvailable: false));

        var devices = BackendRegistry.GetAvailableDevices();

        Assert.Contains(DeviceType.CPU, devices);
        Assert.DoesNotContain(DeviceType.CUDA, devices);
        Assert.Single(devices);
    }

    [Fact]
    public void IsDeviceAvailable_ReturnsTrueForAvailableDevice()
    {
        BackendRegistry.Register(new MockBackend(DeviceType.CPU, isAvailable: true));

        Assert.True(BackendRegistry.IsDeviceAvailable(DeviceType.CPU));
    }

    [Fact]
    public void IsDeviceAvailable_ReturnsFalseForUnavailableDevice()
    {
        BackendRegistry.Register(new MockBackend(DeviceType.CUDA, isAvailable: false));

        Assert.False(BackendRegistry.IsDeviceAvailable(DeviceType.CUDA));
    }

    [Fact]
    public void GetBackend_ReturnsNullForUnregisteredType()
    {
        var backend = BackendRegistry.GetBackend(DeviceType.CUDA);

        Assert.Null(backend);
    }

    [Fact]
    public void Clear_RemovesAllBackends()
    {
        BackendRegistry.Register(new MockBackend(DeviceType.CPU));
        BackendRegistry.Register(new MockBackend(DeviceType.CUDA));

        BackendRegistry.Clear();

        Assert.Null(BackendRegistry.GetBackend(DeviceType.CPU));
        Assert.Null(BackendRegistry.GetBackend(DeviceType.CUDA));
    }

    public BackendRegistryTests()
    {
        BackendRegistry.Clear();
    }
}
