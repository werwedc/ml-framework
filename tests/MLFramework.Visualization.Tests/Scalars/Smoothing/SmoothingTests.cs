using MachineLearning.Visualization.Scalars.Smoothing;

namespace MLFramework.Visualization.Tests.Scalars.Smoothing;

public class SmoothingTests
{
    [Test]
    public void Smooth_WithValidInput_ReturnsSmoothedValues()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        int windowSize = 3;

        // Act
        float[] smoothed = MovingAverageSmoother.Smooth(values, windowSize);

        // Assert
        Assert.That(smoothed.Length, Is.EqualTo(values.Length));
        // Check that smoothed values are different from original (except at boundaries)
        Assert.That(smoothed[2], Is.Not.EqualTo(3.0f).Within(0.01f));
    }

    [Test]
    public void Smooth_WithWindowGreaterThanLength_SmoothsAllValues()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f };
        int windowSize = 10;

        // Act
        float[] smoothed = MovingAverageSmoother.Smooth(values, windowSize);

        // Assert
        Assert.That(smoothed.Length, Is.EqualTo(values.Length));
        // All values should be smoothed to average
        float expectedAverage = 2.0f;
        for (int i = 0; i < smoothed.Length; i++)
        {
            Assert.That(smoothed[i], Is.EqualTo(expectedAverage).Within(0.01f));
        }
    }

    [Test]
    public void Smooth_WithWindowSizeOne_ReturnsOriginalValues()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        int windowSize = 1;

        // Act
        float[] smoothed = MovingAverageSmoother.Smooth(values, windowSize);

        // Assert
        Assert.That(smoothed, Is.EqualTo(values));
    }

    [Test]
    public void Smooth_WithEmptyArray_ReturnsEmptyArray()
    {
        // Arrange
        float[] values = Array.Empty<float>();
        int windowSize = 3;

        // Act
        float[] smoothed = MovingAverageSmoother.Smooth(values, windowSize);

        // Assert
        Assert.That(smoothed, Is.Empty);
    }

    [Test]
    public void Smooth_WithNullInput_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => MovingAverageSmoother.Smooth(null!, 3));
    }

    [Test]
    public void Smooth_WithInvalidWindow_ThrowsArgumentException()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => MovingAverageSmoother.Smooth(values, 0));
        Assert.Throws<ArgumentException>(() => MovingAverageSmoother.Smooth(values, -1));
    }

    [Test]
    public void Smooth_WithConstantValues_ReturnsSameValues()
    {
        // Arrange
        float[] values = { 5.0f, 5.0f, 5.0f, 5.0f, 5.0f };
        int windowSize = 3;

        // Act
        float[] smoothed = MovingAverageSmoother.Smooth(values, windowSize);

        // Assert
        for (int i = 0; i < smoothed.Length; i++)
        {
            Assert.That(smoothed[i], Is.EqualTo(5.0f));
        }
    }

    [Test]
    public void Smooth_WithNoisyData_ReducesNoise()
    {
        // Arrange
        float[] values = { 1.0f, 10.0f, 1.0f, 10.0f, 1.0f, 10.0f, 1.0f };
        int windowSize = 3;

        // Act
        float[] smoothed = MovingAverageSmoother.Smooth(values, windowSize);

        // Assert
        // Calculate variance of smoothed values
        float variance = CalculateVariance(smoothed);
        float originalVariance = CalculateVariance(values);

        // Smoothed variance should be lower than original
        Assert.That(variance, Is.LessThan(originalVariance));
    }

    [Test]
    public void ExponentialSmooth_WithValidInput_ReturnsSmoothedValues()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        float alpha = 0.5f;

        // Act
        float[] smoothed = MovingAverageSmoother.ExponentialSmooth(values, alpha);

        // Assert
        Assert.That(smoothed.Length, Is.EqualTo(values.Length));
        // First value should be the same
        Assert.That(smoothed[0], Is.EqualTo(1.0f));
        // Second value should be 0.5*2.0 + 0.5*1.0 = 1.5
        Assert.That(smoothed[1], Is.EqualTo(1.5f).Within(0.01f));
    }

    [Test]
    public void ExponentialSmooth_WithHighAlpha_FollowsOriginalClosely()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        float alpha = 0.9f;

        // Act
        float[] smoothed = MovingAverageSmoother.ExponentialSmooth(values, alpha);

        // Assert
        // With high alpha, smoothed values should be close to original
        for (int i = 1; i < values.Length; i++)
        {
            Assert.That(smoothed[i], Is.EqualTo(values[i]).Within(0.5f));
        }
    }

    [Test]
    public void ExponentialSmooth_WithLowAlpha_SmoothsMore()
    {
        // Arrange
        float[] values = { 1.0f, 10.0f, 1.0f, 10.0f, 1.0f };
        float alpha = 0.1f;

        // Act
        float[] smoothed = MovingAverageSmoother.ExponentialSmooth(values, alpha);

        // Assert
        // With low alpha, values should change gradually
        for (int i = 1; i < smoothed.Length - 1; i++)
        {
            float difference = Math.Abs(smoothed[i + 1] - smoothed[i]);
            Assert.That(difference, Is.LessThan(2.0f));
        }
    }

    [Test]
    public void ExponentialSmooth_WithEmptyArray_ReturnsEmptyArray()
    {
        // Arrange
        float[] values = Array.Empty<float>();
        float alpha = 0.5f;

        // Act
        float[] smoothed = MovingAverageSmoother.ExponentialSmooth(values, alpha);

        // Assert
        Assert.That(smoothed, Is.Empty);
    }

    [Test]
    public void ExponentialSmooth_WithNullInput_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => MovingAverageSmoother.ExponentialSmooth(null!, 0.5f));
    }

    [Test]
    public void ExponentialSmooth_WithInvalidAlpha_ThrowsArgumentException()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => MovingAverageSmoother.ExponentialSmooth(values, 0.0f));
        Assert.Throws<ArgumentException>(() => MovingAverageSmoother.ExponentialSmooth(values, -0.1f));
        Assert.Throws<ArgumentException>(() => MovingAverageSmoother.ExponentialSmooth(values, 1.5f));
    }

    [Test]
    public void WeightedSmooth_WithUniformWeights_SimilarToMovingAverage()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        float[] weights = { 1.0f, 1.0f, 1.0f }; // Uniform weights

        // Act
        float[] weightedSmoothed = MovingAverageSmoother.WeightedSmooth(values, weights);
        float[] averageSmoothed = MovingAverageSmoother.Smooth(values, 3);

        // Assert
        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(weightedSmoothed[i], Is.EqualTo(averageSmoothed[i]).Within(0.01f));
        }
    }

    [Test]
    public void WeightedSmooth_WithCenterBias_WeightsCenterMore()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 10.0f, 4.0f, 5.0f };
        float[] weights = { 0.1f, 0.8f, 0.1f }; // Strong center bias

        // Act
        float[] smoothed = MovingAverageSmoother.WeightedSmooth(values, weights);

        // Assert
        // Middle value should be heavily weighted
        Assert.That(smoothed[2], Is.GreaterThan(5.0f).And.LessThan(10.0f));
    }

    [Test]
    public void WeightedSmooth_WithEvenLengthWeights_ThrowsArgumentException()
    {
        // Arrange
        float[] values = { 1.0f, 2.0f, 3.0f };
        float[] weights = { 0.5f, 0.5f }; // Even length

        // Act & Assert
        Assert.Throws<ArgumentException>(() => MovingAverageSmoother.WeightedSmooth(values, weights));
    }

    [Test]
    public void WeightedSmooth_WithNullInput_ThrowsArgumentNullException()
    {
        // Arrange
        float[] weights = { 1.0f, 1.0f, 1.0f };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => MovingAverageSmoother.WeightedSmooth(null!, weights));
        Assert.Throws<ArgumentNullException>(() => MovingAverageSmoother.WeightedSmooth(new[] { 1.0f }, null!));
    }

    private float CalculateVariance(float[] values)
    {
        if (values.Length == 0) return 0;
        float mean = values.Average();
        float sumSquaredDiff = values.Sum(v => (v - mean) * (v - mean));
        return sumSquaredDiff / values.Length;
    }
}
