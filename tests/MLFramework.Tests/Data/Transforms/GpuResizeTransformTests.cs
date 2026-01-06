using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data.Transforms;

/// <summary>
/// Tests for GpuResizeTransform.
/// </summary>
public class GpuResizeTransformTests
{
    [Fact]
    public void Constructor_ValidDimensions_CreatesTransform()
    {
        // Arrange & Act
        var transform = new GpuResizeTransform(width: 224, height: 224);

        // Assert
        Assert.NotNull(transform);
    }

    [Fact]
    public void Constructor_NegativeWidth_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new GpuResizeTransform(width: -1, height: 224));
    }

    [Fact]
    public void Constructor_ZeroWidth_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new GpuResizeTransform(width: 0, height: 224));
    }

    [Fact]
    public void Constructor_NegativeHeight_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new GpuResizeTransform(width: 224, height: -1));
    }

    [Fact]
    public void Constructor_ZeroHeight_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new GpuResizeTransform(width: 224, height: 0));
    }

    [Fact]
    public void Apply_FloatArray_ResizesCorrectly()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 2, height: 2);
        var image = new float[,]
        {
            { 1.0f, 2.0f, 3.0f },
            { 4.0f, 5.0f, 6.0f },
            { 7.0f, 8.0f, 9.0f }
        };

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        Assert.Equal(2, result.GetLength(0));
        Assert.Equal(2, result.GetLength(1));
    }

    [Fact]
    public void Apply_FloatArray_ReturnsCorrectDimensions()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 100, height: 100);
        var image = new float[50, 50];

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        Assert.Equal(100, result.GetLength(0));
        Assert.Equal(100, result.GetLength(1));
    }

    [Fact]
    public void Apply_Upscale_IncreasesDimensions()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 10, height: 10);
        var image = new float[5, 5];

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        Assert.Equal(10, result.GetLength(0));
        Assert.Equal(10, result.GetLength(1));
    }

    [Fact]
    public void Apply_Downscale_DecreasesDimensions()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 5, height: 5);
        var image = new float[10, 10];

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        Assert.Equal(5, result.GetLength(0));
        Assert.Equal(5, result.GetLength(1));
    }

    [Fact]
    public void Apply_InvalidInputType_ThrowsException()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 224, height: 224);
        var invalidInput = "not a float array";

        // Act & Assert
        Assert.Throws<ArgumentException>(() => transform.Apply(invalidInput));
    }

    [Fact]
    public void GpuAvailable_ReturnsBoolean()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 224, height: 224);

        // Act & Assert
        var available = transform.GpuAvailable;
        Assert.IsType<bool>(available);
    }

    [Fact]
    public void GpuDevice_ReturnsDefaultZero()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 224, height: 224);

        // Act & Assert
        Assert.Equal(0, transform.GpuDevice);
    }

    [Fact]
    public void SetGpuDevice_UpdatesDeviceId()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 224, height: 224);

        // Act
        transform.SetGpuDevice(2);

        // Assert
        Assert.Equal(2, transform.GpuDevice);
    }

    [Fact]
    public void SetGpuDevice_Zero_UpdatesDeviceId()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 224, height: 224);

        // Act
        transform.SetGpuDevice(0);

        // Assert
        Assert.Equal(0, transform.GpuDevice);
    }

    [Fact]
    public void Apply_SquareImage_SquareOutput()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 64, height: 64);
        var image = new float[32, 32];

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        Assert.Equal(64, result.GetLength(0));
        Assert.Equal(64, result.GetLength(1));
    }

    [Fact]
    public void Apply_RectangularImage_RespectsAspectRatio()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 64, height: 128);
        var image = new float[32, 16];

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        Assert.Equal(128, result.GetLength(0)); // height
        Assert.Equal(64, result.GetLength(1));  // width
    }

    [Fact]
    public void Constructor_DifferentInterpolationModes_AcceptsAllModes()
    {
        // Arrange & Act
        var nearest = new GpuResizeTransform(width: 224, height: 224, mode: InterpolationMode.Nearest);
        var bilinear = new GpuResizeTransform(width: 224, height: 224, mode: InterpolationMode.Bilinear);
        var bicubic = new GpuResizeTransform(width: 224, height: 224, mode: InterpolationMode.Bicubic);
        var lanczos = new GpuResizeTransform(width: 224, height: 224, mode: InterpolationMode.Lanczos);

        // Assert
        Assert.NotNull(nearest);
        Assert.NotNull(bilinear);
        Assert.NotNull(bicubic);
        Assert.NotNull(lanczos);
    }

    [Fact]
    public void Apply_SameSize_PreservesDimensions()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 50, height: 50);
        var image = new float[50, 50];

        // Act
        var result = (float[,])transform.Apply(image);

        // Assert
        Assert.Equal(50, result.GetLength(0));
        Assert.Equal(50, result.GetLength(1));
    }

    [Fact]
    public void Apply_NullInput_ThrowsException()
    {
        // Arrange
        var transform = new GpuResizeTransform(width: 224, height: 224);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => transform.Apply(null));
    }
}
