using RitterFramework.Core;
using RitterFramework.Core.Diagnostics;
using Xunit;

namespace MLFramework.Tests.Diagnostics;

/// <summary>
/// Unit tests for OperationMetadataRegistry class.
/// </summary>
public class OperationMetadataRegistryTests
{
    private IOperationMetadataRegistry _registry;

    public OperationMetadataRegistryTests()
    {
        _registry = new DefaultOperationMetadataRegistry();
    }

    [Fact]
    public void RegisterOperation_RegistersCorrectly()
    {
        // Arrange
        var requirements = new OperationShapeRequirements
        {
            InputCount = 2,
            ExpectedDimensions = new[] { 2, 2 },
            Description = "Test operation"
        };

        // Act
        _registry.RegisterOperation(OperationType.MatrixMultiply, requirements);

        // Assert
        Assert.True(_registry.IsRegistered(OperationType.MatrixMultiply));
        Assert.NotNull(_registry.GetRequirements(OperationType.MatrixMultiply));
    }

    [Fact]
    public void ValidateShapes_MatrixMultiplyValidShapes_ReturnsSuccess()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 10, 5 } };

        // Act
        var result = _registry.ValidateShapes(
            OperationType.MatrixMultiply,
            inputShapes);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void ValidateShapes_MatrixMultiplyInvalidShapes_ReturnsFailure()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 5, 10 } };

        // Act
        var result = _registry.ValidateShapes(
            OperationType.MatrixMultiply,
            inputShapes);

        // Assert
        Assert.False(result.IsValid);
        Assert.NotEmpty(result.Errors);
        Assert.True(result.Errors[0].Contains("mismatch") ||
                        result.Errors[0].Contains("match"));
    }

    [Fact]
    public void ValidateShapes_Conv2DChannelMismatch_ReturnsFailure()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 }, new long[] { 64, 64, 3, 3 } };
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 1, 1 } },
            { "padding", new[] { 0, 0 } }
        };

        // Act
        var result = _registry.ValidateShapes(
            OperationType.Conv2D,
            inputShapes,
            parameters);

        // Assert
        Assert.False(result.IsValid);
    }

    [Fact]
    public void PreRegisteredOperations_AreAvailable()
    {
        // Assert
        Assert.True(_registry.IsRegistered(OperationType.MatrixMultiply));
        Assert.True(_registry.IsRegistered(OperationType.Conv2D));
        Assert.True(_registry.IsRegistered(OperationType.Concat));
    }

    [Fact]
    public void GetRequirements_ReturnsCorrectRequirements()
    {
        // Arrange & Act
        var requirements = _registry.GetRequirements(OperationType.MatrixMultiply);

        // Assert
        Assert.NotNull(requirements);
        Assert.Equal(2, requirements.InputCount);
        Assert.NotNull(requirements.ExpectedDimensions);
        Assert.Equal(2, requirements.ExpectedDimensions.Length);
    }

    [Fact]
    public void GetRequirements_UnregisteredOperation_ReturnsDefaultRequirements()
    {
        // Arrange & Act
        var requirements = _registry.GetRequirements((OperationType)999);

        // Assert
        Assert.NotNull(requirements);
        Assert.Equal("No requirements registered for this operation", requirements.Description);
    }

    [Fact]
    public void IsRegistered_UnregisteredOperation_ReturnsFalse()
    {
        // Act
        var isRegistered = _registry.IsRegistered((OperationType)999);

        // Assert
        Assert.False(isRegistered);
    }

    [Fact]
    public void ValidateShapes_Conv2DValidShapes_ReturnsSuccess()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 }, new long[] { 64, 3, 3, 3 } };
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 1, 1 } },
            { "padding", new[] { 0, 0 } }
        };

        // Act
        var result = _registry.ValidateShapes(
            OperationType.Conv2D,
            inputShapes,
            parameters);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void ValidateShapes_LinearValidShapes_ReturnsSuccess()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 784 }, new long[] { 10, 784 } };

        // Act
        var result = _registry.ValidateShapes(
            OperationType.Linear,
            inputShapes);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void ValidateShapes_LinearInvalidShapes_ReturnsFailure()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 784 }, new long[] { 10, 512 } };

        // Act
        var result = _registry.ValidateShapes(
            OperationType.Linear,
            inputShapes);

        // Assert
        Assert.False(result.IsValid);
        Assert.NotEmpty(result.Errors);
    }

    [Fact]
    public void ValidateShapes_CustomValidator_IsUsed()
    {
        // Arrange
        var customValidatorCalled = false;
        var requirements = new OperationShapeRequirements
        {
            InputCount = 1,
            CustomValidator = (shapes, parameters) =>
            {
                customValidatorCalled = true;
                return ValidationResult.Success();
            }
        };

        _registry.RegisterOperation(OperationType.Conv2D, requirements);
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 } };

        // Act
        var result = _registry.ValidateShapes(
            OperationType.Conv2D,
            inputShapes);

        // Assert
        Assert.True(customValidatorCalled);
        Assert.True(result.IsValid);
    }

    [Fact]
    public void ValidateShapes_WrongInputCount_ReturnsFailure()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 } }; // Only one input

        // Act
        var result = _registry.ValidateShapes(
            OperationType.MatrixMultiply,
            inputShapes);

        // Assert
        Assert.False(result.IsValid);
        Assert.NotEmpty(result.Errors);
        Assert.Contains("Expected", result.Errors[0]);
    }

    [Fact]
    public void ValidateShapes_WrongDimensions_ReturnsFailure()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32 }, new long[] { 10, 5 } }; // First input is 1D

        // Act
        var result = _registry.ValidateShapes(
            OperationType.MatrixMultiply,
            inputShapes);

        // Assert
        Assert.False(result.IsValid);
        Assert.NotEmpty(result.Errors);
    }

    [Fact]
    public void RegisterOperation_OverwritesExistingOperation()
    {
        // Arrange
        var originalRequirements = _registry.GetRequirements(OperationType.MatrixMultiply);
        var newRequirements = new OperationShapeRequirements
        {
            InputCount = 3,
            Description = "New description"
        };

        // Act
        _registry.RegisterOperation(OperationType.MatrixMultiply, newRequirements);
        var updatedRequirements = _registry.GetRequirements(OperationType.MatrixMultiply);

        // Assert
        Assert.Equal(3, updatedRequirements.InputCount);
        Assert.Equal("New description", updatedRequirements.Description);
        Assert.NotEqual(originalRequirements.InputCount, updatedRequirements.InputCount);
    }
}
