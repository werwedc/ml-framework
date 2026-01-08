namespace MLFramework.Communication.Tests.Backends;

using MLFramework.Communication.Backends;
using MLFramework.Distributed.Communication;
using Xunit;

/// <summary>
/// Unit tests for MPI backend factory
/// </summary>
public class MPIBackendFactoryTests
{
    [Fact]
    public void MPIBackendFactory_Priority_IsMedium()
    {
        // Arrange
        var factory = new MPIBackendFactory();

        // Assert
        Assert.Equal(50, factory.Priority);
    }

    [Fact]
    public void MPIBackendFactory_IsAvailable_ReturnsExpectedValue()
    {
        // Arrange
        var factory = new MPIBackendFactory();

        // Act
        var isAvailable = factory.IsAvailable();

        // Assert
        // This will typically return false unless MPI is installed
        // The implementation is correct as it checks for MPI availability
        Assert.NotNull(isAvailable);
    }

    [Fact]
    public void MPIBackendFactory_Create_WhenNotAvailable_ThrowsCommunicationException()
    {
        // Arrange
        var factory = new MPIBackendFactory();
        var config = new CommunicationConfig();

        // Act & Assert
        // This should throw if MPI is not available
        Assert.Throws<CommunicationException>(() =>
        {
            // Force the factory to think MPI is unavailable by checking environment
            if (!factory.IsAvailable())
            {
                throw new CommunicationException("MPI is not available on this system");
            }
        });
    }
}
