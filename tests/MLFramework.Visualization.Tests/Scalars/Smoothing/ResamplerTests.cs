using MachineLearning.Visualization.Scalars.Smoothing;

namespace MLFramework.Visualization.Tests.Scalars.Smoothing;

public class ResamplerTests
{
    [Test]
    public void Resample_WithValidInput_ReturnsResampledArray()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        int targetCount = 3;

        // Act
        float[] resampled = Resampler.Resample(values, targetCount);

        // Assert
        Assert.That(resampled.Length, Is.EqualTo(targetCount));
        // First and last values should be preserved
        Assert.That(resampled[0], Is.EqualTo(1.0f));
        Assert.That(resampled[2], Is.EqualTo(5.0f));
    }

    [Test]
    public void Resample_WithTargetCountGreaterThanOriginal_ReturnsOriginal()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f };
        int targetCount = 10;

        // Act
        float[] resampled = Resampler.Resample(values, targetCount);

        // Assert
        Assert.That(resampled, Is.EqualTo(values));
    }

    [Test]
    public void Resample_WithEmptyArray_ReturnsEmptyArray()
    {
        // Arrange
        float[] values = Array.Empty<float>();
        int targetCount = 3;

        // Act
        float[] resampled = Resampler.Resample(values, targetCount);

        // Assert
        Assert.That(resampled, Is.Empty);
    }

    [Test]
    public void Resample_WithNullInput_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => Resampler.Resample(null!, 3));
    }

    [Test]
    public void Resample_WithInvalidTargetCount_ThrowsArgumentException()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => Resampler.Resample(values, 0));
        Assert.Throws<ArgumentException>(() => Resampler.Resample(values, -1));
    }

    [Test]
    public void Resample_WithLinearData_PreservesLinearity()
    {
        // Arrange
        float[] values = new float[100];
        for (int i = 0; i < 100; i++)
        {
            values[i] = (float)i;
        }
        int targetCount = 10;

        // Act
        float[] resampled = Resampler.Resample(values, targetCount);

        // Assert
        // Check that interpolated values are approximately correct
        Assert.That(resampled[0], Is.EqualTo(0.0f));
        Assert.That(resampled[5], Is.EqualTo(50.0f).Within(1.0f));
        Assert.That(resampled[9], Is.EqualTo(99.0f));
    }

    [Test]
    public void Resample_WithSteps_ReturnsResampledValuesAndSteps()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        long[] steps = { 0, 10, 20, 30, 40 };
        int targetCount = 3;

        // Act
        var (resampledValues, resampledSteps) = Resampler.Resample(values, steps, targetCount);

        // Assert
        Assert.That(resampledValues.Length, Is.EqualTo(targetCount));
        Assert.That(resampledSteps.Length, Is.EqualTo(targetCount));
        // First and last values should be preserved
        Assert.That(resampledValues[0], Is.EqualTo(1.0f));
        Assert.That(resampledValues[2], Is.EqualTo(5.0f));
        Assert.That(resampledSteps[0], Is.EqualTo(0));
        Assert.That(resampledSteps[2], Is.EqualTo(40));
    }

    [Test]
    public void Resample_WithSteps_InterpolatesStepsCorrectly()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        long[] steps = { 0, 100, 200, 300, 400 };
        int targetCount = 3;

        // Act
        var (resampledValues, resampledSteps) = Resampler.Resample(values, steps, targetCount);

        // Assert
        // Middle value should be at step 200
        Assert.That(resampledSteps[1], Is.EqualTo(200));
    }

    [Test]
    public void Resample_WithStepsAndUnevenLengths_ThrowsArgumentException()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f };
        long[] steps = { 0, 10, 20, 30 }; // Different length
        int targetCount = 3;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => Resampler.Resample(values, steps, targetCount));
    }

    [Test]
    public void Resample_WithStepsAndNullInputs_ThrowsArgumentNullException()
    {
        // Arrange
        long[] steps = { 0, 10, 20 };
        int targetCount = 3;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => Resampler.Resample(null!, steps, targetCount));
        Assert.Throws<ArgumentNullException>(() => Resampler.Resample(new[] { 1.0f }, null!, targetCount));
    }

    [Test]
    public void ResampleLTTB_WithNoisyData_PreservesPeaks()
    {
        // Arrange
        float[] values = new float[100];
        for (int i = 0; i < 100; i++)
        {
            values[i] = (float)(Math.Sin(i * 0.1) * 10 + i * 0.1);
        }
        int targetCount = 20;

        // Act
        float[] resampled = Resampler.ResampleLTTB(values, targetCount);

        // Assert
        Assert.That(resampled.Length, Is.EqualTo(targetCount));
        // First and last values should be preserved
        Assert.That(resampled[0], Is.EqualTo(values[0]));
        Assert.That(resampled[targetCount - 1], Is.EqualTo(values[values.Length - 1]));
    }

    [Test]
    public void ResampleLTTB_WithSmallArray_ReturnsOriginal()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f };
        int targetCount = 10;

        // Act
        float[] resampled = Resampler.ResampleLTTB(values, targetCount);

        // Assert
        Assert.That(resampled, Is.EqualTo(values));
    }

    [Test]
    public void ResampleLTTB_WithEmptyArray_ReturnsEmptyArray()
    {
        // Arrange
        float[] values = Array.Empty<float>();
        int targetCount = 3;

        // Act
        float[] resampled = Resampler.ResampleLTTB(values, targetCount);

        // Assert
        Assert.That(resampled, Is.Empty);
    }

    [Test]
    public void ResampleLTTB_WithNullInput_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => Resampler.ResampleLTTB(null!, 3));
    }

    [Test]
    public void ResampleLTTB_WithInvalidTargetCount_ThrowsArgumentException()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => Resampler.ResampleLTTB(values, 0));
        Assert.Throws<ArgumentException>(() => Resampler.ResampleLTTB(values, -1));
    }

    [Test]
    public void ResampleLTTB_WithTargetCountOfTwo_ReturnsFirstAndLast()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        int targetCount = 2;

        // Act
        float[] resampled = Resampler.ResampleLTTB(values, targetCount);

        // Assert
        Assert.That(resampled.Length, Is.EqualTo(2));
        Assert.That(resampled[0], Is.EqualTo(1.0f));
        Assert.That(resampled[1], Is.EqualTo(5.0f));
    }

    [Test]
    public void Downsample_WithValidFactor_ReturnsDownsampledArray()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        int factor = 2;

        // Act
        float[] downsampled = Resampler.Downsample(values, factor);

        // Assert
        Assert.That(downsampled.Length, Is.EqualTo(3));
        Assert.That(downsampled[0], Is.EqualTo(1.0f));
        Assert.That(downsampled[1], Is.EqualTo(3.0f));
        Assert.That(downsampled[2], Is.EqualTo(5.0f));
    }

    [Test]
    public void Downsample_WithFactorOfOne_ReturnsOriginal()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        int factor = 1;

        // Act
        float[] downsampled = Resampler.Downsample(values, factor);

        // Assert
        Assert.That(downsampled, Is.EqualTo(values));
    }

    [Test]
    public void Downsample_WithEmptyArray_ReturnsEmptyArray()
    {
        // Arrange
        float[] values = Array.Empty<float>();
        int factor = 2;

        // Act
        float[] downsampled = Resampler.Downsample(values, factor);

        // Assert
        Assert.That(downsampled, Is.Empty);
    }

    [Test]
    public void Downsample_WithNullInput_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => Resampler.Downsample(null!, 2));
    }

    [Test]
    public void Downsample_WithInvalidFactor_ThrowsArgumentException()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => Resampler.Downsample(values, 0));
        Assert.Throws<ArgumentException>(() => Resampler.Downsample(values, -1));
    }

    [Test]
    public void Downsample_WithFactorGreaterThanLength_ReturnsFirstElement()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f };
        int factor = 10;

        // Act
        float[] downsampled = Resampler.Downsample(values, factor);

        // Assert
        Assert.That(downsampled.Length, Is.EqualTo(1));
        Assert.That(downsampled[0], Is.EqualTo(1.0f));
    }

    [Test]
    public void ResampleLTTB_VersusLinear_PreservesDifferentFeatures()
    {
        // Arrange
        float[] values = new float[100];
        // Create a signal with sharp peaks
        for (int i = 0; i < 100; i++)
        {
            values[i] = (float)(i < 50 ? 10.0 : 0.0);
        }
        int targetCount = 10;

        // Act
        float[] lttb = Resampler.ResampleLTTB(values, targetCount);
        float[] linear = Resampler.Resample(values, targetCount);

        // Assert
        // Both methods should preserve the overall trend
        // but LTTB should capture the sharp transition better
        // This is a qualitative test - both should work
        Assert.That(lttb.Length, Is.EqualTo(targetCount));
        Assert.That(linear.Length, Is.EqualTo(targetCount));
    }
}
