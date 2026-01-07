# Spec: Tests for Device Mesh and Parallelization

## Overview
Create unit tests for the device mesh management and pmap transformation to ensure distributed execution infrastructure works correctly.

## Scope
- Test Device class
- Test DeviceMesh creation and querying
- Test 1D and 2D meshes
- Test pmap basic parallelization
- Test data sharding and gathering

## Test Files to Create
1. `tests/MLFramework.Tests/Functional/DeviceMeshTests.cs`
2. `tests/MLFramework.Tests/Functional/PMapTests.cs`

## Test Requirements - Part 1: DeviceMesh

### 1. Device Tests

```csharp
using Xunit;
using MLFramework.Functional.Distributed;

namespace MLFramework.Tests.Functional
{
    public class DeviceTests
    {
        [Fact]
        public void Constructor_ShouldInitializeCorrectly()
        {
            // Act
            var device = new Device(0, DeviceType.GPU, "gpu:0");

            // Assert
            Assert.Equal(0, device.Id);
            Assert.Equal(DeviceType.GPU, device.Type);
            Assert.Equal("gpu:0", device.Name);
            Assert.True(device.IsAvailable);
        }

        [Fact]
        public void CPU_ShouldCreateCPUDevice()
        {
            // Act
            var device = Device.CPU(5);

            // Assert
            Assert.Equal(5, device.Id);
            Assert.Equal(DeviceType.CPU, device.Type);
            Assert.Equal("cpu:5", device.Name);
        }

        [Fact]
        public void GPU_ShouldCreateGPUDevice()
        {
            // Act
            var device = Device.GPU(3);

            // Assert
            Assert.Equal(3, device.Id);
            Assert.Equal(DeviceType.GPU, device.Type);
            Assert.Equal("gpu:3", device.Name);
        }

        [Fact]
        public void ToString_ShouldReturnCorrectFormat()
        {
            // Arrange
            var device = Device.GPU(0);

            // Act
            var str = device.ToString();

            // Assert
            Assert.Equal("Device(GPU:0)", str);
        }
    }
}
```

### 2. DeviceMesh Tests

```csharp
public class DeviceMeshTests
{
    [Fact]
    public void Create1D_ShouldCreateCorrectMesh()
    {
        // Arrange
        var devices = new[]
        {
            Device.CPU(0), Device.CPU(1), Device.CPU(2), Device.CPU(3)
        };

        // Act
        var mesh = new DeviceMesh(devices);

        // Assert
        Assert.Equal(4, mesh.DeviceCount);
        Assert.Single(mesh.Shape);
        Assert.Equal(4, mesh.Shape[0]);
        Assert.Equal(1, mesh.Rank);
    }

    [Fact]
    public void Create2D_ShouldCreateCorrectMesh()
    {
        // Arrange
        var devices = new Device[6];
        for (int i = 0; i < 6; i++)
            devices[i] = Device.CPU(i);

        // Act
        var mesh = new DeviceMesh(new int[] { 2, 3 }, devices);

        // Assert
        Assert.Equal(6, mesh.DeviceCount);
        Assert.Equal(2, mesh.Rank);
        Assert.Equal(2, mesh.Shape[0]);
        Assert.Equal(3, mesh.Shape[1]);
    }

    [Fact]
    public void Create2D_ShouldThrowForMismatchedDeviceCount()
    {
        // Arrange
        var devices = new Device[4];  // 4 devices but need 6 (2x3)

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new DeviceMesh(new int[] { 2, 3 }, devices));
    }

    [Fact]
    public void GetDevice_1D_ShouldReturnCorrectDevice()
    {
        // Arrange
        var devices = new[] { Device.CPU(0), Device.CPU(1), Device.CPU(2) };
        var mesh = new DeviceMesh(devices);

        // Act
        var device = mesh.GetDevice(1);

        // Assert
        Assert.Same(devices[1], device);
    }

    [Fact]
    public void GetDevice_2D_ShouldReturnCorrectDevice()
    {
        // Arrange
        var devices = new Device[4];
        for (int i = 0; i < 4; i++)
            devices[i] = Device.CPU(i);

        var mesh = new DeviceMesh(new int[] { 2, 2 }, devices);

        // Act
        var device = mesh.GetDevice(0, 1);

        // Assert
        Assert.Same(devices[1], device);
    }

    [Fact]
    public void GetDevice_ShouldThrowForInvalidIndexCount()
    {
        // Arrange
        var devices = new[] { Device.CPU(0), Device.CPU(1) };
        var mesh = new DeviceMesh(devices);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => mesh.GetDevice(0, 0));
    }

    [Fact]
    public void GetDevice_ShouldThrowForOutOfRangeIndex()
    {
        // Arrange
        var devices = new[] { Device.CPU(0), Device.CPU(1) };
        var mesh = new DeviceMesh(devices);

        // Act & Assert
        Assert.Throws<IndexOutOfRangeException>(() => mesh.GetDevice(5));
    }

    [Fact]
    public void ShardingAxis_ShouldReturnCorrectAxis()
    {
        // Arrange
        var devices = new[] { Device.CPU(0), Device.CPU(1) };
        var mesh = new DeviceMesh(devices);

        // Act
        var axis = mesh["data"];

        // Assert
        Assert.NotNull(axis);
        Assert.Equal("data", axis.Name);
    }

    [Fact]
    public void ShardingAxis_ShouldThrowForUnknownAxis()
    {
        // Arrange
        var devices = new[] { Device.CPU(0) };
        var mesh = new DeviceMesh(devices);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
        {
            var _ = mesh["unknown"];
        });
    }
}
```

### 3. DeviceMeshFactory Tests

```csharp
public class DeviceMeshFactoryTests
{
    [Fact]
    public void Create1D_ShouldCreateCPUMesh()
    {
        // Act
        var mesh = DeviceMeshFactory.Create1D(4, DeviceType.CPU);

        // Assert
        Assert.Equal(4, mesh.DeviceCount);
        Assert.Single(mesh.Shape);
        Assert.Equal(4, mesh.Shape[0]);
    }

    [Fact]
    public void Create1D_ShouldCreateGPUMesh()
    {
        // Act
        var mesh = DeviceMeshFactory.Create1D(8, DeviceType.GPU);

        // Assert
        Assert.Equal(8, mesh.DeviceCount);

        // Verify all are GPU devices
        var device = mesh.GetDevice(0);
        Assert.Equal(DeviceType.GPU, device.Type);
    }

    [Fact]
    public void Create2D_ShouldCreateCorrectTopology()
    {
        // Act
        var mesh = DeviceMeshFactory.Create2D(2, 4, DeviceType.CPU);

        // Assert
        Assert.Equal(8, mesh.DeviceCount);
        Assert.Equal(2, mesh.Rank);
        Assert.Equal(2, mesh.Shape[0]);
        Assert.Equal(4, mesh.Shape[1]);
    }

    [Fact]
    public void Default_ShouldReturnSingleCPUMesh()
    {
        // Act
        var mesh = DeviceMeshFactory.Default();

        // Assert
        Assert.Equal(1, mesh.DeviceCount);
        var device = mesh.GetDevice(0);
        Assert.Equal(DeviceType.CPU, device.Type);
    }
}
```

## Test Requirements - Part 2: PMap

### 4. PMap Basic Tests

```csharp
using Xunit;
using MLFramework.Functional;

namespace MLFramework.Tests.Functional
{
    public class PMapTests
{
    [Fact]
    public void Parallelize_SingleInput_ShouldShardAndGather()
    {
        // Arrange
        var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
        Func<Tensor, Tensor> multiplyByTwo = t => t * 2f;
        var parallelMultiply = Functional.Parallelize(multiplyByTwo, mesh);

        // Act
        var input = Tensor.FromArray(new[] { 1f, 2, 3, 4 }).Reshape(4, 1);
        var result = parallelMultiply(input);

        // Assert
        Assert.Equal(new[] { 4, 1 }, result.Shape);
        Assert.Equal(2f, result[0, 0].ToScalar());
        Assert.Equal(4f, result[1, 0].ToScalar());
        Assert.Equal(6f, result[2, 0].ToScalar());
        Assert.Equal(8f, result[3, 0].ToScalar());
    }

    [Fact]
    public void Parallelize_DoubleInput_ShouldShardBothInputs()
    {
        // Arrange
        var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
        Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;
        var parallelAdd = Functional.Parallelize(add, mesh);

        // Act
        var input1 = Tensor.FromArray(new[] { 1f, 2, 3, 4 }).Reshape(4, 1);
        var input2 = Tensor.FromArray(new[] { 10f, 20, 30, 40 }).Reshape(4, 1);
        var result = parallelAdd(input1, input2);

        // Assert
        Assert.Equal(new[] { 4, 1 }, result.Shape);
        Assert.Equal(11f, result[0, 0].ToScalar());
        Assert.Equal(22f, result[1, 0].ToScalar());
        Assert.Equal(33f, result[2, 0].ToScalar());
        Assert.Equal(44f, result[3, 0].ToScalar());
    }

    [Fact]
    public void Parallelize_WithInAxes_ShouldShardOnlyFirstInput()
    {
        // Arrange
        var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
        Func<Tensor, Tensor, Tensor> multiply = (a, b) => a * b;
        var parallelMultiply = Functional.Parallelize(multiply, mesh, new object[] { 0, null });

        // Act
        var batchedInput = Tensor.FromArray(new[] { 1f, 2, 3, 4 }).Reshape(4, 1);
        var singleInput = Tensor.FromArray(new[] { 10f }).Reshape(1);
        var result = parallelMultiply(batchedInput, singleInput);

        // Assert
        Assert.Equal(new[] { 4, 1 }, result.Shape);
        // Each batch element should be multiplied by the same single input
        Assert.Equal(10f, result[0, 0].ToScalar());
        Assert.Equal(20f, result[1, 0].ToScalar());
        Assert.Equal(30f, result[2, 0].ToScalar());
        Assert.Equal(40f, result[3, 0].ToScalar());
    }
}
```

### 5. PMap Error Handling Tests

```csharp
public class PMapTests
{
    [Fact]
    public void Parallelize_ShouldThrowForNullMesh()
    {
        // Arrange
        Func<Tensor, Tensor> identity = t => t;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            Functional.Parallelize(identity, null));
    }

    [Fact]
    public void Parallelize_ShouldThrowForNonDivisibleShard()
    {
        // Arrange
        var mesh = DeviceMeshFactory.Create1D(3, DeviceType.CPU);  // 3 devices
        Func<Tensor, Tensor> identity = t => t;
        var parallelIdentity = Functional.Parallelize(identity, mesh);

        var input = Tensor.FromArray(new float[5]).Reshape(5, 1);  // 5 elements (not divisible by 3)

        // Act & Assert
        Assert.Throws<ArgumentException>(() => parallelIdentity(input));
    }

    [Fact]
    public void Parallelize_ShouldThrowForInvalidInAxesLength()
    {
        // Arrange
        var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
        Func<Tensor, Tensor, Tensor> add = (a, b) => a + b;

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            Functional.Parallelize(add, mesh, new object[] { 0 }));  // Should have 2 axes
    }

    [Fact]
    public void Parallelize_ShouldThrowForNonTensorReturn()
    {
        // Arrange
        var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
        Func<Tensor, int> getCount = t => t.Shape.TotalElements;

        // Act & Assert
        Assert.Throws<NotSupportedException>(() =>
            Functional.Parallelize(getCount, mesh));
    }
}
```

### 6. PMap Real-World Scenario Tests

```csharp
public class PMapTests
{
    [Fact]
    public void Parallelize_BatchTrainingStep()
    {
        // Arrange
        var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
        Func<Tensor, Tensor> forwardPass = batch =>
        {
            // Simulate forward pass
            return batch.MatMul(Tensor.FromArray(new float[12]).Reshape(4, 3));
        };

        var parallelForward = Functional.Parallelize(forwardPass, mesh);

        // Act
        var input = Tensor.FromArray(new float[24]).Reshape(8, 3);  // Batch of 8
        var result = parallelForward(input);

        // Assert
        Assert.Equal(new[] { 8, 3 }, result.Shape);
    }

    [Fact]
    public void Parallelize_WithWeights_ShouldNotShardWeights()
    {
        // Arrange
        var mesh = DeviceMeshFactory.Create1D(2, DeviceType.CPU);
        var weights = Tensor.FromArray(new float[12]).Reshape(4, 3);

        Func<Tensor, Tensor, Tensor> applyWeights = (input, w) => input.MatMul(w);
        var parallelApply = Functional.Parallelize(applyWeights, mesh, new object[] { 0, null });

        // Act
        var input = Tensor.FromArray(new float[24]).Reshape(8, 4);
        var result = parallelApply(input, weights);

        // Assert
        Assert.Equal(new[] { 8, 3 }, result.Shape);
    }
}
```

## Files to Create
1. `tests/MLFramework.Tests/Functional/DeviceMeshTests.cs`
2. `tests/MLFramework.Tests/Functional/PMapTests.cs`

## Dependencies
- spec_functional_device_mesh.md (implementation must be complete)
- spec_functional_pmap_basic.md (implementation must be complete)

## Success Criteria
- All tests pass
- DeviceMesh creation and querying work correctly
- PMap sharding and gathering work correctly
- Error cases are properly tested
- Mixed sharding scenarios work

## Notes for Coder
- Use xUnit for testing
- Device execution is simulated (not actual multi-device) for now
- Test that sharding preserves data integrity
- Test edge cases: single device, many devices, non-divisible batch sizes
- Verify that the original function is not modified
- Consider testing with 2D meshes for more complex scenarios
- Test that pmap and vmap can be composed together

## Additional Considerations
- Add performance tests to measure overhead of sharding/gathering
- Test thread safety if multiple threads use the same mesh
- Consider integration tests with actual device execution when available
- Test cache invalidation when device configuration changes
