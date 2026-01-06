using MLFramework.Data.Transforms;
using Xunit;

namespace MLFramework.Tests.Data.Transforms;

public class ToTensorTransformTests
{
    [Fact]
    public void Apply_ShouldConvertInputToTensorPlaceholder()
    {
        // Arrange
        var transform = new ToTensorTransform();
        var input = new float[,]
        {
            { 1.0f, 2.0f, 3.0f },
            { 4.0f, 5.0f, 6.0f }
        };

        // Act
        var result = transform.Apply(input);

        // Assert
        Assert.NotNull(result);
        var tensor = Assert.IsType<ToTensorTransform.TensorPlaceholder>(result);
        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(3, tensor.Shape[1]);
        Assert.Equal(input, tensor.Data);
    }

    [Fact]
    public void Apply_ShouldHandleEmptyArray()
    {
        // Arrange
        var transform = new ToTensorTransform();
        var input = new float[0, 0];

        // Act
        var result = transform.Apply(input);

        // Assert
        Assert.NotNull(result);
        var tensor = Assert.IsType<ToTensorTransform.TensorPlaceholder>(result);
        Assert.Equal(0, tensor.Shape[0]);
        Assert.Equal(0, tensor.Shape[1]);
    }

    [Fact]
    public void Apply_ShouldHandleLargeArray()
    {
        // Arrange
        var transform = new ToTensorTransform();
        var input = new float[100, 100];
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                input[i, j] = i * 100 + j;
            }
        }

        // Act
        var result = transform.Apply(input);

        // Assert
        Assert.NotNull(result);
        var tensor = Assert.IsType<ToTensorTransform.TensorPlaceholder>(result);
        Assert.Equal(100, tensor.Shape[0]);
        Assert.Equal(100, tensor.Shape[1]);
        Assert.Equal(input, tensor.Data);
    }

    [Fact]
    public void Apply_WithNullInput_ShouldThrowArgumentNullException()
    {
        // Arrange
        var transform = new ToTensorTransform();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => transform.Apply(null));
    }
}
