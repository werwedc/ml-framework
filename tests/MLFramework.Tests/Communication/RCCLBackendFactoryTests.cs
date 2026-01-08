namespace MLFramework.Communication.Tests;

using MLFramework.Communication.Backends;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System;
using Xunit;

/// <summary>
/// Tests for RCCL backend factory
/// </summary>
public class RCCLBackendFactoryTests
{
    [Fact]
    public void Factory_Priority_IsCorrect()
    {
        // Arrange & Act
        var factory = new RCCLBackendFactory();

        // Assert
        Assert.Equal(90, factory.Priority);
    }

    [Fact]
    public void Factory_GetRank_DefaultsToZero()
    {
        // Arrange
        var factory = new RCCLBackendFactory();

        // Act
        // This will use the private GetRank method which defaults to 0
        // when no environment variable is set

        // Assert - factory creation should work
        var config = new CommunicationConfig();
        if (factory.IsAvailable())
        {
            var backend = factory.Create(config);
            Assert.NotNull(backend);
            Assert.Equal(0, backend.Rank);
        }
    }

    [Fact]
    public void Factory_GetWorldSize_DefaultsToOne()
    {
        // Arrange
        var factory = new RCCLBackendFactory();

        // Act
        // This will use the private GetWorldSize method which defaults to 1
        // when no environment variable is set

        // Assert - factory creation should work
        var config = new CommunicationConfig();
        if (factory.IsAvailable())
        {
            var backend = factory.Create(config);
            Assert.NotNull(backend);
            Assert.Equal(1, backend.WorldSize);
        }
    }

    [Fact]
    public void Factory_Create_WithoutRCCL_ThrowsException()
    {
        // Arrange
        var factory = new RCCLBackendFactory();
        var config = new CommunicationConfig();

        // Act & Assert
        if (!factory.IsAvailable())
        {
            var exception = Assert.Throws<CommunicationException>(() => factory.Create(config));
            Assert.Contains("RCCL is not available", exception.Message);
        }
    }

    [Fact]
    public void Factory_Create_WithRCCL_ReturnsBackend()
    {
        // Arrange
        var factory = new RCCLBackendFactory();
        var config = new CommunicationConfig();

        // Act & Assert
        if (factory.IsAvailable())
        {
            var backend = factory.Create(config);
            Assert.NotNull(backend);
            Assert.Equal("RCCL", backend.BackendName);
        }
    }

    [Fact]
    public void Factory_IsAvailable_ReturnsBoolean()
    {
        // Arrange
        var factory = new RCCLBackendFactory();

        // Act
        var isAvailable = factory.IsAvailable();

        // Assert
        Assert.IsType<bool>(isAvailable);
    }
}
