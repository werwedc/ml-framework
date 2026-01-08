namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using MLFramework.Communication.Operations;
using RitterFramework.Core.Tensor;
using Xunit;

/// <summary>
/// Unit tests for scatter and gather helper operations
/// </summary>
public class ScatterGatherHelpersTests
{
    [Fact]
    public void Scatter_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        ICommunicationBackend backend = null;
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3, 4 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ScatterGatherHelpers.Scatter(backend, tensor));
    }

    [Fact]
    public void Scatter_WithNullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        Tensor tensor = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ScatterGatherHelpers.Scatter(backend, tensor));
    }

    [Fact]
    public void Scatter_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 });

        // Act
        var result = ScatterGatherHelpers.Scatter(backend, tensor);

        // Assert
        Assert.NotNull(result);
        // Should have 1/4th of the elements (2 out of 8)
        Assert.Equal(2, result.Size);
    }

    [Fact]
    public void Gather_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        ICommunicationBackend backend = null;
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ScatterGatherHelpers.Gather(backend, tensor));
    }

    [Fact]
    public void Gather_WithNullTensor_ThrowsArgumentNullException()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        Tensor tensor = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ScatterGatherHelpers.Gather(backend, tensor));
    }

    [Fact]
    public void Gather_WithValidParameters_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var result = ScatterGatherHelpers.Gather(backend, tensor);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(tensor.Size, result.Size);
    }

    [Fact]
    public void Gather_WithRootRank_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);
        var tensor = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var result = ScatterGatherHelpers.Gather(backend, tensor, 0);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(tensor.Size, result.Size);
    }
}
