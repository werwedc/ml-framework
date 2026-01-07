namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using MLFramework.Communication.Operations;
using RitterFramework.Core.Tensor;
using Xunit;

/// <summary>
/// Unit tests for synchronous all-reduce operation
/// </summary>
public class AllReduceTests
{
    [Fact]
    public void AllReduceTensor_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        ICommunicationBackend backend = null;
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            AllReduce.AllReduceTensor(backend, tensor, ReduceOp.Sum));
    }

    [Fact]
    public void AllReduceTensor_WithNullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        Tensor tensor = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            AllReduce.AllReduceTensor(backend, tensor, ReduceOp.Sum));
    }

    [Fact]
    public void AllReduceTensor_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var result = AllReduce.AllReduceTensor(backend, tensor, ReduceOp.Sum);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(tensor.Size, result.Size);
    }

    [Fact]
    public void AverageGradients_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var gradients = Tensor.FromArray(new float[] { 4, 8, 12 });

        // Act
        var result = AllReduce.AverageGradients(backend, gradients);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(gradients.Size, result.Size);
        // After dividing by 4: [1, 2, 3]
        Assert.Equal(1.0f, result.Data[0]);
        Assert.Equal(2.0f, result.Data[1]);
        Assert.Equal(3.0f, result.Data[2]);
    }

    [Fact]
    public void AllReduceTensors_WithMultipleTensors_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensors = new[]
        {
            Tensor.FromArray(new float[] { 1, 2, 3 }),
            Tensor.FromArray(new float[] { 4, 5, 6 })
        };

        // Act
        var results = AllReduce.AllReduceTensors(backend, tensors, ReduceOp.Sum);

        // Assert
        Assert.NotNull(results);
        Assert.Equal(2, results.Count);
    }
}
