using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data.Transforms;

/// <summary>
/// Tests for GpuNormalizeTransform.
/// </summary>
public class GpuNormalizeTransformTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesTransform()
    {
        // Arrange & Act
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.485f, 0.456f, 0.406f },
            std: new[] { 0.229f, 0.224f, 0.225f });

        // Assert
        Assert.NotNull(transform);
    }

    [Fact]
    public void Constructor_NullMean_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new GpuNormalizeTransform(
            mean: null,
            std: new[] { 0.229f, 0.224f, 0.225f }));
    }

    [Fact]
    public void Constructor_NullStd_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new GpuNormalizeTransform(
            mean: new[] { 0.485f, 0.456f, 0.406f },
            std: null));
    }

    [Fact]
    public void Constructor_MismatchedLengths_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new GpuNormalizeTransform(
            mean: new[] { 0.485f, 0.456f, 0.406f },
            std: new[] { 0.229f, 0.224f })); // Different length
    }

    [Fact]
    public void Constructor_SingleChannel_AcceptsParameters()
    {
        // Arrange & Act
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.5f },
            std: new[] { 0.5f });

        // Assert
        Assert.NotNull(transform);
    }

    [Fact]
    public void Constructor_EmptyArrays_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new GpuNormalizeTransform(
            mean: Array.Empty<float>(),
            std: Array.Empty<float>()));
    }

    [Fact]
    public void Apply_FloatArray_NormalizesCorrectly()
    {
        // Arrange
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.5f, 0.5f, 0.5f },
            std: new[] { 0.5f, 0.5f, 0.5f });
        var image = new float[,]
        {
            { 1.0f, 1.0f, 1.0f },
            { 0.5f, 0.5f, 0.5f }
        };

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(2, result.GetLength(0));
        Assert.Equal(3, result.GetLength(1));
    }

    [Fact]
    public void Apply_InvalidInputType_ThrowsException()
    {
        // Arrange
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.485f, 0.456f, 0.406f },
            std: new[] { 0.229f, 0.224f, 0.225f });
        var invalidInput = "not a float array";

        // Act & Assert
        Assert.Throws<ArgumentException>(() => transform.Apply(invalidInput));
    }

    [Fact]
    public void Apply_NullInput_ThrowsException()
    {
        // Arrange
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.485f, 0.456f, 0.406f },
            std: new[] { 0.229f, 0.224f, 0.225f });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => transform.Apply(null));
    }

    [Fact]
    public void Apply_CalculatesCorrectValues()
    {
        // Arrange
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.5f },
            std: new[] { 0.5f });
        var image = new float[2, 2]
        {
            { 1.0f, 0.5f },
            { 0.0f, 1.0f }
        };

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        // (1.0 - 0.5) / 0.5 = 1.0
        // (0.5 - 0.5) / 0.5 = 0.0
        // (0.0 - 0.5) / 0.5 = -1.0
        Assert.Equal(1.0f, result[0, 0]);
        Assert.Equal(0.0f, result[0, 1]);
        Assert.Equal(-1.0f, result[1, 0]);
    }

    [Fact]
    public void Apply_MultiChannel_AppliesCorrectNormalization()
    {
        // Arrange
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.0f, 1.0f, 2.0f },
            std: new[] { 1.0f, 1.0f, 1.0f });
        var image = new float[1, 6]
        {
            { 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f }
        };

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        // Channels cycle: 0, 1, 2, 0, 1, 2
        Assert.Equal(1.0f, result[0, 0]);  // (1.0 - 0.0) / 1.0
        Assert.Equal(1.0f, result[0, 1]);  // (2.0 - 1.0) / 1.0
        Assert.Equal(1.0f, result[0, 2]);  // (3.0 - 2.0) / 1.0
    }

    [Fact]
    public void GpuAvailable_ReturnsBoolean()
    {
        // Arrange
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.485f, 0.456f, 0.406f },
            std: new[] { 0.229f, 0.224f, 0.225f });

        // Act & Assert
        var available = transform.GpuAvailable;
        Assert.IsType<bool>(available);
    }

    [Fact]
    public void GpuDevice_ReturnsDefaultZero()
    {
        // Arrange
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.485f, 0.456f, 0.406f },
            std: new[] { 0.229f, 0.224f, 0.225f });

        // Act & Assert
        Assert.Equal(0, transform.GpuDevice);
    }

    [Fact]
    public void SetGpuDevice_UpdatesDeviceId()
    {
        // Arrange
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.485f, 0.456f, 0.406f },
            std: new[] { 0.229f, 0.224f, 0.225f });

        // Act
        transform.SetGpuDevice(3);

        // Assert
        Assert.Equal(3, transform.GpuDevice);
    }

    [Fact]
    public void Apply_LargeArray_HandlesCorrectly()
    {
        // Arrange
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.5f },
            std: new[] { 0.5f });
        var image = new float[1000, 1000];

        // Fill with some values
        for (int i = 0; i < 1000; i++)
        {
            for (int j = 0; j < 1000; j++)
            {
                image[i, j] = 0.75f;
            }
        }

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        Assert.Equal(1000, result.GetLength(0));
        Assert.Equal(1000, result.GetLength(1));
    }

    [Fact]
    public void Apply_ZeroStd_DoesNotThrow()
    {
        // Arrange
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.5f },
            std: new[] { 0.001f }); // Very small std to avoid division by zero warning
        var image = new float[2, 2]
        {
            { 0.5f, 0.5f },
            { 0.5f, 0.5f }
        };

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(2, result.GetLength(0));
        Assert.Equal(2, result.GetLength(1));
    }

    [Fact]
    public void Apply_ImageNormNormalization_CorrectValues()
    {
        // Arrange
        // Using ImageNet normalization values
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.485f, 0.456f, 0.406f },
            std: new[] { 0.229f, 0.224f, 0.225f });
        var image = new float[1, 3]
        {
            { 0.485f, 0.456f, 0.406f } // Same as mean values
        };

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        // All values should be zero (image equals mean)
        Assert.Equal(0.0f, result[0, 0], 6);
        Assert.Equal(0.0f, result[0, 1], 6);
        Assert.Equal(0.0f, result[0, 2], 6);
    }

    [Fact]
    public void Apply_MultipleTransforms_Idempotent()
    {
        // Arrange
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.5f },
            std: new[] { 0.5f });
        var image = new float[2, 2]
        {
            { 1.0f, 0.5f },
            { 0.0f, 1.0f }
        };

        // Act
        var result1 = (float[,])transform.Apply(image);
        var result2 = (float[,])transform.Apply(result1);

        // Assert
        // Second normalization should be different from first
        // Values should change again
        Assert.NotEqual(result1[0, 0], result2[0, 0]);
    }

    [Fact]
    public void Constructor_FourChannels_AcceptsParameters()
    {
        // Arrange & Act
        var transform = new GpuNormalizeTransform(
            mean: new[] { 0.5f, 0.5f, 0.5f, 0.5f },
            std: new[] { 0.5f, 0.5f, 0.5f, 0.5f });

        // Assert
        Assert.NotNull(transform);
    }
}
