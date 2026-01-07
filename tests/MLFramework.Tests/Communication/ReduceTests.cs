namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using MLFramework.Communication.Operations;
using RitterFramework.Core.Tensor;
using Xunit;

/// <summary>
/// Unit tests for synchronous reduce operation
/// </summary>
public class ReduceTests
{
    [Fact]
    public void ReduceTensor_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        ICommunicationBackend backend = null;
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            Reduce.ReduceTensor(backend, tensor, ReduceOp.Sum, 0));
    }

    [Fact]
    public void ReduceTensor_WithNullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        Tensor tensor = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            Reduce.ReduceTensor(backend, tensor, ReduceOp.Sum, 0));
    }

    [Fact]
    public void ReduceTensor_WithInvalidRootRank_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => 
            Reduce.ReduceTensor(backend, tensor, ReduceOp.Sum, -1));
        
        Assert.Throws<ArgumentOutOfRangeException>(() => 
            Reduce.ReduceTensor(backend, tensor, ReduceOp.Sum, 4));
    }

    [Fact]
    public void ReduceTensor_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var result = Reduce.ReduceTensor(backend, tensor, ReduceOp.Sum, 0);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(tensor.Size, result.Size);
    }

    [Fact]
    public void ReduceTensor_WithProductAndBoolType_ThrowsArgumentException()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = new Tensor(new float[] { 1, 2, 3 }, new int[] { 3 }, false, DataType.Bool);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => 
            Reduce.ReduceTensor(backend, tensor, ReduceOp.Product, 0));
    }

    [Fact]
    public void ReduceTensors_WithMultipleTensors_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensors = new[]
        {
            Tensor.FromArray(new float[] { 1, 2, 3 }),
            Tensor.FromArray(new float[] { 4, 5, 6 })
        };

        // Act
        var results = Reduce.ReduceTensors(backend, tensors, ReduceOp.Sum, 0);

        // Assert
        Assert.NotNull(results);
        Assert.Equal(2, results.Count);
    }
}
