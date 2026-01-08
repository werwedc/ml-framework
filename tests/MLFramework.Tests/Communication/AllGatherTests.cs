namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using MLFramework.Communication.Operations;
using RitterFramework.Core.Tensor;
using Xunit;

/// <summary>
/// Unit tests for synchronous all-gather operation
/// </summary>
public class AllGatherTests
{
    [Fact]
    public void AllGatherTensor_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        ICommunicationBackend backend = null;
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            AllGather.AllGatherTensor(backend, tensor));
    }

    [Fact]
    public void AllGatherTensor_WithNullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        Tensor tensor = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            AllGather.AllGatherTensor(backend, tensor));
    }

    [Fact]
    public void AllGatherTensor_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var result = AllGather.AllGatherTensor(backend, tensor);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(tensor.Size, result.Size);
    }

    [Fact]
    public void AllGatherTensors_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var result = AllGather.AllGatherTensors(backend, tensor);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(4, result.Count); // Should get 4 tensors (one from each rank)
    }

    [Fact]
    public void SplitGatheredTensor_WithValidTensor_SplitsCorrectly()
    {
        // Arrange
        var data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var tensor = Tensor.FromArray(data);
        int worldSize = 4;

        // Act
        var result = AllGather.SplitGatheredTensor(tensor, worldSize);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(4, result.Count);
        Assert.Equal(2, result[0].Size); // Each chunk should have 2 elements
        Assert.Equal(2, result[1].Size);
        Assert.Equal(2, result[2].Size);
        Assert.Equal(2, result[3].Size);
    }
}
