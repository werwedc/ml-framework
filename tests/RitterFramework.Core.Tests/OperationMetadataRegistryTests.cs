using Xunit;
using RitterFramework.Core;
using RitterFramework.Core.Operations;

namespace RitterFramework.Tests;

public class OperationMetadataRegistryTests
{
    #region Singleton Pattern Tests

    [Fact]
    public void Instance_ReturnsSameInstance()
    {
        // Act
        var instance1 = OperationMetadataRegistry.Instance;
        var instance2 = OperationMetadataRegistry.Instance;

        // Assert
        Assert.Same(instance1, instance2);
    }

    #endregion

    #region Registration Tests

    [Fact]
    public void Register_ValidMetadata_SuccessfullyRegisters()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var metadata = new MatrixMultiplyMetadata();

        // Act
        registry.Register(OperationType.MatrixMultiply, metadata);

        // Assert
        Assert.True(registry.IsRegistered(OperationType.MatrixMultiply));
        Assert.Same(metadata, registry.GetMetadata(OperationType.MatrixMultiply));
    }

    [Fact]
    public void Register_NullMetadata_ThrowsArgumentNullException()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => registry.Register(OperationType.Linear, null!));
    }

    [Fact]
    public void Register_TypeMismatch_ThrowsArgumentException()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var matrixMultiplyMetadata = new MatrixMultiplyMetadata();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => registry.Register(OperationType.Conv2D, matrixMultiplyMetadata));
    }

    [Fact]
    public void IsRegistered_RegisteredOperation_ReturnsTrue()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;

        // Act & Assert
        Assert.True(registry.IsRegistered(OperationType.MatrixMultiply));
        Assert.True(registry.IsRegistered(OperationType.Conv2D));
        Assert.True(registry.IsRegistered(OperationType.Concat));
    }

    [Fact]
    public void IsRegistered_UnregisteredOperation_ReturnsFalse()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;

        // Act & Assert
        Assert.False(registry.IsRegistered(OperationType.Linear));
        Assert.False(registry.IsRegistered(OperationType.Stack));
    }

    #endregion

    #region Retrieval Tests

    [Fact]
    public void GetMetadata_RegisteredOperation_ReturnsMetadata()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;

        // Act
        var metadata = registry.GetMetadata(OperationType.MatrixMultiply);

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(OperationType.MatrixMultiply, metadata.Type);
        Assert.Equal("MatrixMultiply", metadata.Name);
    }

    [Fact]
    public void GetMetadata_UnregisteredOperation_ThrowsKeyNotFoundException()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;

        // Act & Assert
        Assert.Throws<KeyNotFoundException>(() => registry.GetMetadata(OperationType.Linear));
    }

    #endregion

    #region MatrixMultiply Validation Tests

    [Fact]
    public void Validate_MatrixMultiply_ValidShapes_ReturnsValid()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var shape1 = new long[] { 32, 256 };   // [M, K]
        var shape2 = new long[] { 256, 10 };   // [K, N]

        // Act
        var result = registry.Validate(OperationType.MatrixMultiply, shape1, shape2);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.ErrorMessage);
    }

    [Fact]
    public void Validate_MatrixMultiply_InvalidInnerDimensions_ReturnsInvalid()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var shape1 = new long[] { 32, 256 };   // [M, K]
        var shape2 = new long[] { 128, 10 };   // [K', N] - K != K'

        // Act
        var result = registry.Validate(OperationType.MatrixMultiply, shape1, shape2);

        // Assert
        Assert.False(result.IsValid);
        Assert.NotEmpty(result.ErrorMessage);
    }

    [Fact]
    public void Validate_MatrixMultiply_WrongNumberOfInputs_ReturnsInvalid()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var shape1 = new long[] { 32, 256 };

        // Act
        var result = registry.Validate(OperationType.MatrixMultiply, shape1);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("requires 2 input tensors", result.ErrorMessage);
    }

    [Fact]
    public void InferOutputShape_MatrixMultiply_ReturnsCorrectShape()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var shape1 = new long[] { 32, 256 };   // [M, K]
        var shape2 = new long[] { 256, 10 };   // [K, N]

        // Act
        var outputShape = registry.InferOutputShape(OperationType.MatrixMultiply, shape1, shape2);

        // Assert
        Assert.Equal(2, outputShape.Length);
        Assert.Equal(32, outputShape[0]);  // M
        Assert.Equal(10, outputShape[1]);  // N
    }

    #endregion

    #region Conv2D Validation Tests

    [Fact]
    public void Validate_Conv2D_ValidShapes_ReturnsValid()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var inputShape = new long[] { 32, 3, 224, 224 };   // [N, C, H, W]
        var kernelShape = new long[] { 64, 3, 3, 3 };      // [F, C, kH, kW]

        // Act
        var result = registry.Validate(OperationType.Conv2D, inputShape, kernelShape);

        // Assert
        Assert.True(result.IsValid);
    }

    [Fact]
    public void Validate_Conv2D_ChannelMismatch_ReturnsInvalid()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var inputShape = new long[] { 32, 3, 224, 224 };   // [N, C, H, W]
        var kernelShape = new long[] { 64, 4, 3, 3 };      // [F, C', kH, kW] - C != C'

        // Act
        var result = registry.Validate(OperationType.Conv2D, inputShape, kernelShape);

        // Assert
        Assert.False(result.IsValid);
    }

    [Fact]
    public void InferOutputShape_Conv2D_ReturnsCorrectShape()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var inputShape = new long[] { 32, 3, 224, 224 };   // [N, C, H, W]
        var kernelShape = new long[] { 64, 3, 3, 3 };      // [F, C, kH, kW]

        // Act
        var outputShape = registry.InferOutputShape(OperationType.Conv2D, inputShape, kernelShape);

        // Assert
        Assert.Equal(4, outputShape.Length);
        Assert.Equal(32, outputShape[0]);  // N (batch)
        Assert.Equal(64, outputShape[1]);  // F (filters)
        Assert.Equal(222, outputShape[2]);  // H_out = (224 - 3)/1 + 1
        Assert.Equal(222, outputShape[3]);  // W_out = (224 - 3)/1 + 1
    }

    [Fact]
    public void InferOutputShape_Conv2D_WithPadding_ReturnsCorrectShape()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var inputShape = new long[] { 32, 3, 224, 224 };
        var kernelShape = new long[] { 64, 3, 3, 3 };

        // Register custom Conv2D with padding
        registry.Register(OperationType.Conv2D, new Conv2DMetadata(stride: 1, padding: 1));

        // Act
        var outputShape = registry.InferOutputShape(OperationType.Conv2D, inputShape, kernelShape);

        // Assert
        Assert.Equal(4, outputShape.Length);
        Assert.Equal(32, outputShape[0]);
        Assert.Equal(64, outputShape[1]);
        Assert.Equal(224, outputShape[2]);  // H_out = (224 + 2 - 3)/1 + 1
        Assert.Equal(224, outputShape[3]);  // W_out = (224 + 2 - 3)/1 + 1
    }

    #endregion

    #region Concat Validation Tests

    [Fact]
    public void Validate_Concat_ValidShapes_ReturnsValid()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var shape1 = new long[] { 32, 128 };
        var shape2 = new long[] { 32, 64 };

        // Act
        var result = registry.Validate(OperationType.Concat, shape1, shape2);

        // Assert
        Assert.True(result.IsValid);
    }

    [Fact]
    public void Validate_Concat_DifferentDimensions_ReturnsInvalid()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var shape1 = new long[] { 32, 128 };
        var shape2 = new long[] { 16, 64 };  // Different batch dimension

        // Act
        var result = registry.Validate(OperationType.Concat, shape1, shape2);

        // Assert
        Assert.False(result.IsValid);
    }

    [Fact]
    public void InferOutputShape_Concat_ReturnsCorrectShape()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;
        var shape1 = new long[] { 32, 128 };
        var shape2 = new long[] { 32, 64 };

        // Act
        var outputShape = registry.InferOutputShape(OperationType.Concat, shape1, shape2);

        // Assert
        Assert.Equal(2, outputShape.Length);
        Assert.Equal(32, outputShape[0]);   // Batch size
        Assert.Equal(192, outputShape[1]);  // 128 + 64
    }

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void Validate_NullInputShapes_ReturnsInvalid()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;

        // Act
        var result = registry.Validate(OperationType.MatrixMultiply, null!);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("cannot be null", result.ErrorMessage);
    }

    [Fact]
    public void Validate_EmptyInputShapes_ReturnsInvalid()
    {
        // Arrange
        var registry = OperationMetadataRegistry.Instance;

        // Act
        var result = registry.Validate(OperationType.MatrixMultiply, Array.Empty<long[]>());

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("cannot be null or empty", result.ErrorMessage);
    }

    #endregion

    #region Metadata Interface Tests

    [Fact]
    public void MatrixMultiplyMetadata_TypeProperty_ReturnsCorrectType()
    {
        // Arrange
        var metadata = new MatrixMultiplyMetadata();

        // Act & Assert
        Assert.Equal(OperationType.MatrixMultiply, metadata.Type);
    }

    [Fact]
    public void MatrixMultiplyMetadata_NameProperty_ReturnsCorrectName()
    {
        // Arrange
        var metadata = new MatrixMultiplyMetadata();

        // Act & Assert
        Assert.Equal("MatrixMultiply", metadata.Name);
    }

    [Fact]
    public void MatrixMultiplyMetadata_RequiredInputTensors_ReturnsTwo()
    {
        // Arrange
        var metadata = new MatrixMultiplyMetadata();

        // Act & Assert
        Assert.Equal(2, metadata.RequiredInputTensors);
    }

    [Fact]
    public void Conv2DMetadata_TypeProperty_ReturnsCorrectType()
    {
        // Arrange
        var metadata = new Conv2DMetadata();

        // Act & Assert
        Assert.Equal(OperationType.Conv2D, metadata.Type);
    }

    [Fact]
    public void ConcatMetadata_TypeProperty_ReturnsCorrectType()
    {
        // Arrange
        var metadata = new ConcatMetadata();

        // Act & Assert
        Assert.Equal(OperationType.Concat, metadata.Type);
    }

    [Fact]
    public void ConcatMetadata_CustomAxis_ReturnsCorrectName()
    {
        // Arrange
        var metadata = new ConcatMetadata(axis: 1);

        // Act & Assert
        Assert.Equal("Concat (axis=1)", metadata.Name);
        Assert.Equal(1, metadata.Axis);
    }

    #endregion
}
