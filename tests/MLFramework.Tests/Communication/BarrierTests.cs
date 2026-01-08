namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using MLFramework.Communication.Operations;
using Xunit;

/// <summary>
/// Unit tests for barrier synchronization operation
/// </summary>
public class BarrierTests
{
    [Fact]
    public void Synchronize_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        ICommunicationBackend backend = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            Barrier.Synchronize(backend));
    }

    [Fact]
    public void Synchronize_WithValidBackend_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);

        // Act & Assert
        Barrier.Synchronize(backend); // Should not throw
    }

    [Fact]
    public void TrySynchronize_WithNullBackend_ThrowsArgumentNullException()
    {
        // Arrange
        ICommunicationBackend backend = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            Barrier.TrySynchronize(backend, 1000));
    }

    [Fact]
    public void TrySynchronize_WithValidBackend_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);

        // Act
        var result = Barrier.TrySynchronize(backend, 1000);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void TrySynchronize_WithShortTimeout_Succeeds()
    {
        // Arrange
        var backend = new MockCommunicationBackend(0, 4);

        // Act
        var result = Barrier.TrySynchronize(backend, 100);

        // Assert
        Assert.True(result);
    }
}
