using MLFramework.Data.Transforms;
using Xunit;

namespace MLFramework.Tests.Data.Transforms;

public class LambdaTransformTests
{
    [Fact]
    public void Constructor_WithNullFunc_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new LambdaTransform<int, int>(null));
    }

    [Fact]
    public void Constructor_WithValidFunc_ShouldCreateTransform()
    {
        // Arrange
        Func<int, int> func = x => x * 2;

        // Act
        var transform = new LambdaTransform<int, int>(func);

        // Assert
        Assert.NotNull(transform);
    }

    [Fact]
    public void Apply_ShouldExecuteProvidedFunction()
    {
        // Arrange
        Func<int, int> func = x => x * 2;
        var transform = new LambdaTransform<int, int>(func);

        // Act
        var result = transform.Apply(5);

        // Assert
        Assert.Equal(10, result);
    }

    [Fact]
    public void Apply_WithAdditionFunction_ShouldWork()
    {
        // Arrange
        Func<int, int> func = x => x + 10;
        var transform = new LambdaTransform<int, int>(func);

        // Act
        var result = transform.Apply(5);

        // Assert
        Assert.Equal(15, result);
    }

    [Fact]
    public void Apply_WithComplexFunction_ShouldWork()
    {
        // Arrange
        Func<int, int> func = x => (x * x) + (2 * x) + 1;
        var transform = new LambdaTransform<int, int>(func);

        // Act
        var result = transform.Apply(3);

        // Assert
        Assert.Equal(16, result); // 3*3 + 2*3 + 1 = 9 + 6 + 1 = 16
    }

    [Fact]
    public void Apply_WithStringFunction_ShouldWork()
    {
        // Arrange
        Func<int, string> func = x => $"Value: {x}";
        var transform = new LambdaTransform<int, string>(func);

        // Act
        var result = transform.Apply(42);

        // Assert
        Assert.Equal("Value: 42", result);
    }

    [Fact]
    public void Apply_WithArrayFunction_ShouldWork()
    {
        // Arrange
        Func<int[], int[]> func = arr => arr.Select(x => x * 2).ToArray();
        var transform = new LambdaTransform<int[], int[]>(func);
        var input = new[] { 1, 2, 3, 4, 5 };

        // Act
        var result = transform.Apply(input);

        // Assert
        Assert.Equal(new[] { 2, 4, 6, 8, 10 }, result);
    }

    [Fact]
    public void Apply_WithArray2DFunction_ShouldWork()
    {
        // Arrange
        Func<float[,], float[,]> func = arr =>
        {
            int height = arr.GetLength(0);
            int width = arr.GetLength(1);
            var result = new float[height, width];

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    result[i, j] = arr[i, j] * 2.0f;
                }
            }

            return result;
        };

        var transform = new LambdaTransform<float[,], float[,]>(func);
        var input = new float[,]
        {
            { 1.0f, 2.0f },
            { 3.0f, 4.0f }
        };

        // Act
        var result = transform.Apply(input);

        // Assert
        Assert.Equal(2.0f, result[0, 0], 5);
        Assert.Equal(4.0f, result[0, 1], 5);
        Assert.Equal(6.0f, result[1, 0], 5);
        Assert.Equal(8.0f, result[1, 1], 5);
    }

    [Fact]
    public void Apply_WithIdentityFunction_ShouldReturnSameValue()
    {
        // Arrange
        Func<int, int> func = x => x;
        var transform = new LambdaTransform<int, int>(func);

        // Act
        var result = transform.Apply(42);

        // Assert
        Assert.Equal(42, result);
    }

    [Fact]
    public void Apply_WithConditionalFunction_ShouldWork()
    {
        // Arrange
        Func<int, string> func = x => x > 0 ? "positive" : x < 0 ? "negative" : "zero";
        var transform = new LambdaTransform<int, string>(func);

        // Act & Assert
        Assert.Equal("positive", transform.Apply(5));
        Assert.Equal("negative", transform.Apply(-3));
        Assert.Equal("zero", transform.Apply(0));
    }

    [Fact]
    public void Apply_MultipleCalls_ShouldWorkConsistently()
    {
        // Arrange
        Func<int, int> func = x => x * 3;
        var transform = new LambdaTransform<int, int>(func);

        // Act
        var result1 = transform.Apply(1);
        var result2 = transform.Apply(2);
        var result3 = transform.Apply(3);

        // Assert
        Assert.Equal(3, result1);
        Assert.Equal(6, result2);
        Assert.Equal(9, result3);
    }
}
