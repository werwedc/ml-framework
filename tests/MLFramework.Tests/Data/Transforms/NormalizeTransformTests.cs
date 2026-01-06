using MLFramework.Data.Transforms;
using Xunit;

namespace MLFramework.Tests.Data.Transforms;

public class NormalizeTransformTests
{
    [Fact]
    public void Constructor_WithNullMean_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new NormalizeTransform(null, new float[] { 1.0f }));
    }

    [Fact]
    public void Constructor_WithNullStd_ShouldThrowArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new NormalizeTransform(new float[] { 0.5f }, null));
    }

    [Fact]
    public void Constructor_WithMismatchedLengths_ShouldThrowArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new NormalizeTransform(new float[] { 0.5f, 0.5f }, new float[] { 1.0f }));
    }

    [Fact]
    public void Constructor_WithValidParameters_ShouldCreateTransform()
    {
        // Act
        var transform = new NormalizeTransform(new float[] { 0.5f }, new float[] { 0.5f });

        // Assert
        Assert.NotNull(transform);
    }

    [Fact]
    public void Apply_ShouldApplyNormalizationCorrectly()
    {
        // Arrange
        var mean = new float[] { 0.5f };
        var std = new float[] { 0.5f };
        var transform = new NormalizeTransform(mean, std);
        var input = new float[,]
        {
            { 1.0f, 2.0f },
            { 0.0f, 0.5f }
        };

        // Act
        var result = transform.Apply(input);

        // Assert
        Assert.Equal((1.0f - 0.5f) / 0.5f, result[0, 0], 5);
        Assert.Equal((2.0f - 0.5f) / 0.5f, result[0, 1], 5);
        Assert.Equal((0.0f - 0.5f) / 0.5f, result[1, 0], 5);
        Assert.Equal((0.5f - 0.5f) / 0.5f, result[1, 1], 5);
    }

    [Fact]
    public void Apply_WithMultipleChannels_ShouldMapCorrectly()
    {
        // Arrange
        var mean = new float[] { 0.5f, 0.5f, 0.5f };
        var std = new float[] { 0.5f, 0.5f, 0.5f };
        var transform = new NormalizeTransform(mean, std);
        var input = new float[,]
        {
            { 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f }
        };

        // Act
        var result = transform.Apply(input);

        // Assert
        // Channels should cycle: 0, 1, 2, 0, 1, 2
        Assert.Equal((1.0f - 0.5f) / 0.5f, result[0, 0], 5); // Channel 0
        Assert.Equal((1.0f - 0.5f) / 0.5f, result[0, 1], 5); // Channel 1
        Assert.Equal((1.0f - 0.5f) / 0.5f, result[0, 2], 5); // Channel 2
        Assert.Equal((0.0f - 0.5f) / 0.5f, result[0, 3], 5); // Channel 0
        Assert.Equal((0.0f - 0.5f) / 0.5f, result[0, 4], 5); // Channel 1
        Assert.Equal((0.0f - 0.5f) / 0.5f, result[0, 5], 5); // Channel 2
    }

    [Fact]
    public void Apply_ShouldHandleZeroStd()
    {
        // Arrange
        var mean = new float[] { 0.5f };
        var std = new float[] { 0.5f };
        var transform = new NormalizeTransform(mean, std);
        var input = new float[,]
        {
            { 0.5f, 0.5f },
            { 0.5f, 0.5f }
        };

        // Act
        var result = transform.Apply(input);

        // Assert
        Assert.Equal(0.0f, result[0, 0], 5);
        Assert.Equal(0.0f, result[0, 1], 5);
        Assert.Equal(0.0f, result[1, 0], 5);
        Assert.Equal(0.0f, result[1, 1], 5);
    }

    [Fact]
    public void Apply_WithNullInput_ShouldThrowArgumentNullException()
    {
        // Arrange
        var transform = new NormalizeTransform(new float[] { 0.5f }, new float[] { 0.5f });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => transform.Apply(null));
    }

    [Fact]
    public void Apply_ShouldHandleLargeArray()
    {
        // Arrange
        var mean = new float[] { 0.5f };
        var std = new float[] { 0.5f };
        var transform = new NormalizeTransform(mean, std);
        var input = new float[100, 100];
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                input[i, j] = 1.0f;
            }
        }

        // Act
        var result = transform.Apply(input);

        // Assert
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                Assert.Equal((1.0f - 0.5f) / 0.5f, result[i, j], 5);
            }
        }
    }

    [Fact]
    public void Apply_ShouldHandleNegativeValues()
    {
        // Arrange
        var mean = new float[] { 0.0f };
        var std = new float[] { 1.0f };
        var transform = new NormalizeTransform(mean, std);
        var input = new float[,]
        {
            { -1.0f, 0.0f, 1.0f }
        };

        // Act
        var result = transform.Apply(input);

        // Assert
        Assert.Equal(-1.0f, result[0, 0], 5);
        Assert.Equal(0.0f, result[0, 1], 5);
        Assert.Equal(1.0f, result[0, 2], 5);
    }
}
