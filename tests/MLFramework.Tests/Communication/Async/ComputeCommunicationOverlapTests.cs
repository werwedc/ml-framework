using MLFramework.Communication.Async;
using MLFramework.Communication;
using RitterFramework.Core.Tensor;
using System;
using System.Threading.Tasks;
using Xunit;
using Moq;

namespace MLFramework.Tests.Communication.Async;

public class ComputeCommunicationOverlapTests
{
    [Fact]
    public void StartAllReduce_WithValidParameters_ReturnsHandle()
    {
        // Arrange
        var backendMock = new Mock<IAsyncCommunicationBackend>();
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });
        backendMock.Setup(b => b.AllReduce(tensor, ReduceOp.Sum))
                  .Returns(Task.FromResult(tensor));

        // Act
        var handle = ComputeCommunicationOverlap.StartAllReduce(backendMock.Object, tensor, ReduceOp.Sum);

        // Assert
        Assert.NotNull(handle);
    }

    [Fact]
    public void ComputeWhileCommunicating_WithValidParameters_ExecutesComputeDuringComm()
    {
        // Arrange
        var backendMock = new Mock<IAsyncCommunicationBackend>();
        var commTensor = Tensor.FromArray(new float[] { 1.0f, 2.0f, 3.0f });
        var computeTensor = Tensor.FromArray(new float[] { 4.0f, 5.0f, 6.0f });
        
        backendMock.Setup(b => b.AllReduce(commTensor, ReduceOp.Sum))
                  .Returns(Task.FromResult(commTensor));

        var computeCalled = false;
        Func<Tensor> computeFunc = () =>
        {
            computeCalled = true;
            return computeTensor;
        };

        // Act
        var result = ComputeCommunicationOverlap.ComputeWhileCommunicating(
            backendMock.Object, commTensor, computeFunc, ReduceOp.Sum);

        // Assert
        Assert.NotNull(result);
        Assert.True(computeCalled);
    }

    [Fact]
    public void ComputeWhileCommunicating_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        var tensor = Tensor.FromArray(new float[] { 1.0f, 2.0f });
        Func<Tensor> computeFunc = () => tensor;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ComputeCommunicationOverlap.ComputeWhileCommunicating(null, tensor, computeFunc));
    }

    [Fact]
    public void PipelineComputeCommunicate_WithValidParameters_ExecutesPipeline()
    {
        // Arrange
        var backendMock = new Mock<IAsyncCommunicationBackend>();
        var tensors = new List<Tensor>
        {
            Tensor.FromArray(new float[] { 1.0f, 2.0f }),
            Tensor.FromArray(new float[] { 3.0f, 4.0f })
        };

        backendMock.Setup(b => b.AllReduce(It.IsAny<Tensor>(), ReduceOp.Sum))
                  .Returns((Tensor t, ReduceOp op) => Task.FromResult(t));

        Func<Tensor, Tensor> computeFunc = t => t;

        // Act
        var results = ComputeCommunicationOverlap.PipelineComputeCommunicate(
            backendMock.Object, tensors, computeFunc, ReduceOp.Sum);

        // Assert
        Assert.Equal(2, results.Count);
        backendMock.Verify(b => b.AllReduce(It.IsAny<Tensor>(), ReduceOp.Sum), Times.Exactly(2));
    }
}
