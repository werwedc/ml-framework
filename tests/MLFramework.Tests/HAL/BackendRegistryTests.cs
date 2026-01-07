using MLFramework.HAL;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.HAL;

/// <summary>
/// Tests for BackendRegistry class
/// </summary>
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

    [Fact]
    public void Register_AddsBackend()
    {
        var backend = new MockBackend(DeviceType.CPU);
        BackendRegistry.Register(backend);

        var retrieved = BackendRegistry.GetBackend(DeviceType.CPU);
        Assert.Same(backend, retrieved);
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
    public void GetAvailableDevices_ReturnsRegisteredTypes()
    {
        BackendRegistry.Register(new MockBackend(DeviceType.CPU));
        BackendRegistry.Register(new MockBackend(DeviceType.CUDA));

        var devices = BackendRegistry.GetAvailableDevices();

        Assert.Contains(DeviceType.CPU, devices);
        Assert.Contains(DeviceType.CUDA, devices);
    }

    [Fact]
    public void GetBackend_ReturnsNullForUnregisteredType()
    {
        var backend = BackendRegistry.GetBackend(DeviceType.CUDA);

        Assert.Null(backend);
    }

    [Fact]
    public void IsDeviceAvailable_ReturnsTrueForRegisteredAvailableBackend()
    {
        BackendRegistry.Register(new MockBackend(DeviceType.CPU));

        Assert.True(BackendRegistry.IsDeviceAvailable(DeviceType.CPU));
    }

    [Fact]
    public void IsDeviceAvailable_ReturnsFalseForUnregisteredBackend()
    {
        Assert.False(BackendRegistry.IsDeviceAvailable(DeviceType.CUDA));
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

    [Fact]
    public void Register_WithNullBackend_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
        {
            BackendRegistry.Register(null!);
        });
    }

    public BackendRegistryTests()
    {
        BackendRegistry.Clear();
    }
}
