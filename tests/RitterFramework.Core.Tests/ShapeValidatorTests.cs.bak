using RitterFramework.Core;
using MLFramework.Core;
using MLFramework.Shapes;

namespace RitterFramework.Tests;

/// <summary>
/// Unit tests for ShapeValidator class.
/// </summary>
public class ShapeValidatorTests
{
    [Fact]
    public void ValidateMatrixMultiply_ValidShapes_ReturnsValid()
    {
        // Arrange
        long[] shape1 = [32, 256];
        long[] shape2 = [256, 10];

        // Act
        var result = ShapeValidator.ValidateMatrixMultiply(shape1, shape2);

        // Assert
        Assert.True(result.IsValid, "Should be valid for compatible matrix multiplication shapes");
    }

    [Fact]
    public void ValidateMatrixMultiply_IncompatibleInnerDimensions_ReturnsInvalid()
    {
        // Arrange
        long[] shape1 = [32, 256];
        long[] shape2 = [128, 10];

        // Act
        var result = ShapeValidator.ValidateMatrixMultiply(shape1, shape2);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for incompatible inner dimensions");
        Assert.True(result.ErrorMessage.Contains("256"), "Error message should mention first inner dimension");
        Assert.True(result.ErrorMessage.Contains("128"), "Error message should mention second inner dimension");
        Assert.True(result.SuggestedFixes.Count > 0, "Should provide suggested fixes");
    }

    [Fact]
    public void ValidateMatrixMultiply_Non2DShape1_ReturnsInvalid()
    {
        // Arrange
        long[] shape1 = [32, 256, 10];
        long[] shape2 = [256, 10];

        // Act
        var result = ShapeValidator.ValidateMatrixMultiply(shape1, shape2);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for non-2D first tensor");
        Assert.True(result.ErrorMessage.Contains("2-dimensional"), "Error should mention 2D requirement");
    }

    [Fact]
    public void ValidateMatrixMultiply_Non2DShape2_ReturnsInvalid()
    {
        // Arrange
        long[] shape1 = [32, 256];
        long[] shape2 = [256, 10, 5];

        // Act
        var result = ShapeValidator.ValidateMatrixMultiply(shape1, shape2);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for non-2D second tensor");
        Assert.True(result.ErrorMessage.Contains("2-dimensional"), "Error should mention 2D requirement");
    }

    [Fact]
    public void ValidateMatrixMultiply_NullShape1_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ShapeValidator.ValidateMatrixMultiply(null!, [256, 10]));
    }

    [Fact]
    public void ValidateMatrixMultiply_NullShape2_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ShapeValidator.ValidateMatrixMultiply([32, 256], null!));
    }

    [Fact]
    public void ValidateConv2D_ValidShapes_ReturnsValid()
    {
        // Arrange
        long[] inputShape = [32, 3, 224, 224];
        long[] kernelShape = [64, 3, 3, 3];
        int stride = 1;
        int padding = 0;

        // Act
        var result = ShapeValidator.ValidateConv2D(inputShape, kernelShape, stride, padding);

        // Assert
        Assert.True(result.IsValid, "Should be valid for compatible convolution shapes");
    }

    [Fact]
    public void ValidateConv2D_ChannelMismatch_ReturnsInvalid()
    {
        // Arrange
        long[] inputShape = [32, 3, 224, 224];
        long[] kernelShape = [64, 5, 3, 3]; // Wrong channel count

        // Act
        var result = ShapeValidator.ValidateConv2D(inputShape, kernelShape, stride: 1, padding: 0);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for channel mismatch");
        Assert.True(result.ErrorMessage.Contains("channels"), "Error should mention channels");
        Assert.True(result.SuggestedFixes.Count > 0, "Should provide suggested fixes");
    }

    [Fact]
    public void ValidateConv2D_KernelExceedsInput_ReturnsInvalid()
    {
        // Arrange
        long[] inputShape = [32, 3, 10, 10];
        long[] kernelShape = [64, 3, 20, 20]; // Kernel larger than input

        // Act
        var result = ShapeValidator.ValidateConv2D(inputShape, kernelShape, stride: 1, padding: 0);

        // Assert
        Assert.False(result.IsValid, "Should be invalid when kernel exceeds input dimensions");
        Assert.True(result.ErrorMessage.Contains("exceeds"), "Error should mention kernel exceeding input");
    }

    [Fact]
    public void ValidateConv2D_InvalidStride_ReturnsInvalid()
    {
        // Arrange
        long[] inputShape = [32, 3, 224, 224];
        long[] kernelShape = [64, 3, 3, 3];
        int stride = 0;

        // Act
        var result = ShapeValidator.ValidateConv2D(inputShape, kernelShape, stride, padding: 0);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for stride less than 1");
    }

    [Fact]
    public void ValidateConv2D_InvalidPadding_ReturnsInvalid()
    {
        // Arrange
        long[] inputShape = [32, 3, 224, 224];
        long[] kernelShape = [64, 3, 3, 3];
        int padding = -1;

        // Act
        var result = ShapeValidator.ValidateConv2D(inputShape, kernelShape, stride: 1, padding);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for negative padding");
    }

    [Fact]
    public void ValidateConv2D_Non4DInput_ReturnsInvalid()
    {
        // Arrange
        long[] inputShape = [32, 3, 224];
        long[] kernelShape = [64, 3, 3, 3];

        // Act
        var result = ShapeValidator.ValidateConv2D(inputShape, kernelShape, stride: 1, padding: 0);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for non-4D input");
    }

    [Fact]
    public void ValidateConv2D_Non4DKernel_ReturnsInvalid()
    {
        // Arrange
        long[] inputShape = [32, 3, 224, 224];
        long[] kernelShape = [64, 3, 3];

        // Act
        var result = ShapeValidator.ValidateConv2D(inputShape, kernelShape, stride: 1, padding: 0);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for non-4D kernel");
    }

    [Fact]
    public void ValidateConcat_ValidShapes_ReturnsValid()
    {
        // Arrange
        var inputShapes = new List<long[]>
        {
            [32, 128],
            [32, 64]
        };
        int axis = 1;

        // Act
        var result = ShapeValidator.ValidateConcat(inputShapes, axis);

        // Assert
        Assert.True(result.IsValid, "Should be valid for compatible concatenation shapes");
    }

    [Fact]
    public void ValidateConcat_DimensionMismatch_ReturnsInvalid()
    {
        // Arrange
        var inputShapes = new List<long[]>
        {
            [32, 128],
            [32, 64],
            [32, 128, 5] // Different number of dimensions
        };
        int axis = 1;

        // Act
        var result = ShapeValidator.ValidateConcat(inputShapes, axis);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for different number of dimensions");
    }

    [Fact]
    public void ValidateConcat_NonAxisDimensionMismatch_ReturnsInvalid()
    {
        // Arrange
        var inputShapes = new List<long[]>
        {
            [32, 128],
            [64, 64] // Dimension 0 (batch size) differs
        };
        int axis = 1;

        // Act
        var result = ShapeValidator.ValidateConcat(inputShapes, axis);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for non-concat axis dimension mismatch");
    }

    [Fact]
    public void ValidateConcat_EmptyList_ReturnsInvalid()
    {
        // Arrange
        var inputShapes = new List<long[]>();
        int axis = 1;

        // Act
        var result = ShapeValidator.ValidateConcat(inputShapes, axis);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for empty input list");
    }

    [Fact]
    public void ValidateConcat_AxisOutOfBounds_ReturnsInvalid()
    {
        // Arrange
        var inputShapes = new List<long[]>
        {
            [32, 128],
            [32, 64]
        };
        int axis = 5; // Out of bounds

        // Act
        var result = ShapeValidator.ValidateConcat(inputShapes, axis);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for out-of-bounds axis");
    }

    [Fact]
    public void ValidateBroadcast_CompatibleShapes_ReturnsValid()
    {
        // Arrange
        long[] shape1 = [32, 1];
        long[] shape2 = [32, 10];

        // Act
        var result = ShapeValidator.ValidateBroadcast(shape1, shape2);

        // Assert
        Assert.True(result.IsValid, "Should be valid for broadcast-compatible shapes");
    }

    [Fact]
    public void ValidateBroadcast_SameShapes_ReturnsValid()
    {
        // Arrange
        long[] shape1 = [32, 10];
        long[] shape2 = [32, 10];

        // Act
        var result = ShapeValidator.ValidateBroadcast(shape1, shape2);

        // Assert
        Assert.True(result.IsValid, "Should be valid for identical shapes");
    }

    [Fact]
    public void ValidateBroadcast_ScalarBroadcast_ReturnsValid()
    {
        // Arrange
        long[] shape1 = [1];
        long[] shape2 = [32, 10];

        // Act
        var result = ShapeValidator.ValidateBroadcast(shape1, shape2);

        // Assert
        Assert.True(result.IsValid, "Should be valid for scalar broadcast");
    }

    [Fact]
    public void ValidateBroadcast_IncompatibleShapes_ReturnsInvalid()
    {
        // Arrange
        long[] shape1 = [32, 10];
        long[] shape2 = [20, 10];

        // Act
        var result = ShapeValidator.ValidateBroadcast(shape1, shape2);

        // Assert
        Assert.False(result.IsValid, "Should be invalid for incompatible shapes");
        Assert.True(result.SuggestedFixes.Count > 0, "Should provide suggested fixes");
    }

    [Fact]
    public void ValidateBroadcast_NullShape1_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ShapeValidator.ValidateBroadcast(null!, [32, 10]));
    }

    [Fact]
    public void ValidateBroadcast_NullShape2_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ShapeValidator.ValidateBroadcast([32, 10], null!));
    }

    [Fact]
    public void Validate_GenericMatrixMultiply_ReturnsValid()
    {
        // Arrange
        long[] shape1 = [32, 256];
        long[] shape2 = [256, 10];

        // Act
        var result = ShapeValidator.Validate(OperationType.MatrixMultiply, shape1, shape2);

        // Assert
        Assert.True(result.IsValid);
    }

    [Fact]
    public void Validate_GenericConv2D_ReturnsValid()
    {
        // Arrange
        long[] inputShape = [32, 3, 224, 224];
        long[] kernelShape = [64, 3, 3, 3];

        // Act
        var result = ShapeValidator.Validate(OperationType.Conv2D, inputShape, kernelShape);

        // Assert
        Assert.True(result.IsValid);
    }

    [Fact]
    public void Validate_GenericConcat_ReturnsValid()
    {
        // Arrange
        var inputShapes = new long[][]
        {
            [32, 128],
            [32, 64]
        };

        // Act
        var result = ShapeValidator.Validate(OperationType.Concat, inputShapes);

        // Assert
        Assert.True(result.IsValid);
    }

    [Fact]
    public void Validate_GenericBroadcast_ReturnsValid()
    {
        // Arrange
        long[] shape1 = [32, 1];
        long[] shape2 = [32, 10];

        // Act
        var result = ShapeValidator.Validate(OperationType.Broadcast, shape1, shape2);

        // Assert
        Assert.True(result.IsValid);
    }
}

/// <summary>
/// Unit tests for ShapeValidationHelper class.
/// </summary>
[TestClass]
public class ShapeValidationHelperTests
{
    [Fact]
    public void CreateException_FromInvalidResult_ReturnsException()
    {
        // Arrange
        var validationResult = ValidationResult.Invalid("Test error", new List<string> { "Fix 1", "Fix 2" });
        var operationType = OperationType.MatrixMultiply;
        long[][] inputShapes = [[32, 256], [128, 10]];

        // Act
        var exception = ShapeValidationHelper.CreateException(
            validationResult,
            operationType,
            "TestLayer",
            inputShapes);

        // Assert
        Assert.IsNotNull(exception);
        Assert.IsInstanceOfType<ShapeMismatchException>(exception);
        Assert.True(exception.Message.Contains("TestLayer"), "Exception should include layer name");
        Assert.True(exception.Message.Contains("Test error"), "Exception should include error message");
        Assert.True(exception.Message.Contains("Fix 1"), "Exception should include suggested fix 1");
        Assert.True(exception.Message.Contains("Fix 2"), "Exception should include suggested fix 2");
    }

    [Fact]
    public void CreateException_FromValidResult_ThrowsArgumentException()
    {
        // Arrange
        var validationResult = ValidationResult.Valid();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            ShapeValidationHelper.CreateException(validationResult, OperationType.MatrixMultiply));
    }

    [Fact]
    public void BuildProblemDescription_MatrixMultiply_ReturnsCorrectDescription()
    {
        // Arrange
        long[][] inputShapes = [[32, 256], [256, 10]];

        // Act
        var description = ShapeValidationHelper.BuildProblemDescription(
            OperationType.MatrixMultiply, inputShapes);

        // Assert
        Assert.True(description.Contains("Matrix multiplication"));
        Assert.True(description.Contains("[32, 256]"));
        Assert.True(description.Contains("[256, 10]"));
    }

    [Fact]
    public void BuildProblemDescription_Conv2D_ReturnsCorrectDescription()
    {
        // Arrange
        long[][] inputShapes = [[32, 3, 224, 224], [64, 3, 3, 3]];

        // Act
        var description = ShapeValidationHelper.BuildProblemDescription(
            OperationType.Conv2D, inputShapes);

        // Assert
        Assert.True(description.Contains("Conv2D"));
        Assert.True(description.Contains("[32, 3, 224, 224]"));
        Assert.True(description.Contains("[64, 3, 3, 3]"));
    }

    [Fact]
    public void BuildProblemDescription_Concat_ReturnsCorrectDescription()
    {
        // Arrange
        long[][] inputShapes = [[32, 128], [32, 64]];

        // Act
        var description = ShapeValidationHelper.BuildProblemDescription(
            OperationType.Concat, inputShapes);

        // Assert
        Assert.True(description.Contains("Concatenation"));
        Assert.True(description.Contains("[32, 128]"));
        Assert.True(description.Contains("[32, 64]"));
    }

    [Fact]
    public void BuildProblemDescription_Broadcast_ReturnsCorrectDescription()
    {
        // Arrange
        long[][] inputShapes = [[32, 1], [32, 10]];

        // Act
        var description = ShapeValidationHelper.BuildProblemDescription(
            OperationType.Broadcast, inputShapes);

        // Assert
        Assert.True(description.Contains("Broadcast"));
        Assert.True(description.Contains("[32, 1]"));
        Assert.True(description.Contains("[32, 10]"));
    }

    [Fact]
    public void BuildProblemDescription_EmptyShapes_ReturnsGenericDescription()
    {
        // Arrange
        long[][] inputShapes = [];

        // Act
        var description = ShapeValidationHelper.BuildProblemDescription(
            OperationType.MatrixMultiply, inputShapes);

        // Assert
        Assert.True(description.Contains("MatrixMultiply"));
    }
}
