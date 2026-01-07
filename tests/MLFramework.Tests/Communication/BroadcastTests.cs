namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using MLFramework.Communication.Operations;
using MLFramework.Communication.Operations.Async;
using RitterFramework.Core.Tensor;
using Xunit;

/// <summary>
/// Unit tests for synchronous broadcast operation
/// </summary>
public class BroadcastTests
{
    [Fact]
    public void BroadcastTensor_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        ICommunicationBackend backend = null;
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            Broadcast.BroadcastTensor(backend, tensor, 0));
    }

    [Fact]
    public void BroadcastTensor_WithNullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        Tensor tensor = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            Broadcast.BroadcastTensor(backend, tensor, 0));
    }

    [Fact]
    public void BroadcastTensor_WithInvalidRootRank_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => 
            Broadcast.BroadcastTensor(backend, tensor, -1));
        
        Assert.Throws<ArgumentOutOfRangeException>(() => 
            Broadcast.BroadcastTensor(backend, tensor, 4));
    }

    [Fact]
    public void BroadcastTensor_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });
        int rootRank = 0;

        // Act & Assert - should not throw
        var exception = Record.Exception(() => 
            Broadcast.BroadcastTensor(backend, tensor, rootRank));
        
        Assert.Null(exception);
    }

    [Fact]
    public void BroadcastTensors_WithMultipleTensors_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensors = new[]
        {
            Tensor.FromArray(new float[] { 1, 2, 3 }),
            Tensor.FromArray(new float[] { 4, 5, 6 }),
            Tensor.FromArray(new float[] { 7, 8, 9 })
        };

        // Act & Assert - should not throw
        var exception = Record.Exception(() => 
            Broadcast.BroadcastTensors(backend, tensors, 0));
        
        Assert.Null(exception);
    }
}
