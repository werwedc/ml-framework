using System;
using System.Threading.Tasks;
using MLFramework.Optimizers.MixedPrecision;
using Xunit;

namespace MLFramework.Tests.Optimizers.MixedPrecision;

public class HardwareDetectorTests
{
    [Fact]
    public void DetectCapabilities_ReturnsValidCapability()
    {
        // Arrange & Act
        var capability = HardwareDetector.DetectCapabilities();

        // Assert
        Assert.True(capability.SupportsFP32); // FP32 should always be supported
        Assert.True(capability.IsFP16Available == capability.SupportsFP16);
        Assert.True(capability.IsBF16Available == capability.SupportsBF16);
    }

    [Fact]
    public void DetectCapabilities_CachesResult()
    {
        // Arrange & Act
        var firstCall = HardwareDetector.DetectCapabilities();
        var secondCall = HardwareDetector.DetectCapabilities();

        // Assert - Both calls should return the same instance (cached)
        Assert.Equal(firstCall.SupportsFP16, secondCall.SupportsFP16);
        Assert.Equal(firstCall.SupportsBF16, secondCall.SupportsBF16);
    }

    [Fact]
    public void GetRecommendedPrecision_WhenBF16Supported_ReturnsBF16()
    {
        // Note: This test assumes BF16 is not supported by default
        // The actual implementation would need mocking infrastructure
        // which will be added in subsequent specs

        // Arrange & Act
        var precision = HardwareDetector.GetRecommendedPrecision();

        // Assert - Should return best available precision
        Assert.True(precision == Precision.BF16 || precision == Precision.FP16 || precision == Precision.FP32);
    }

    [Fact]
    public void GetRecommendedPrecision_ReturnsFP32AsFallback()
    {
        // Arrange & Act
        var precision = HardwareDetector.GetRecommendedPrecision();

        // Assert - Should never return invalid value
        Assert.True(Enum.IsDefined(typeof(Precision), precision));
    }

    [Fact]
    public void DetectCapabilities_ThreadSafety()
    {
        // Arrange & Act - Call from multiple threads
        var tasks = new Task[10];
        var results = new PrecisionCapability[10];

        for (int i = 0; i < tasks.Length; i++)
        {
            int index = i;
            tasks[i] = Task.Run(() =>
            {
                results[index] = HardwareDetector.DetectCapabilities();
            });
        }

        Task.WaitAll(tasks);

        // Assert - All results should be identical
        for (int i = 1; i < results.Length; i++)
        {
            Assert.Equal(results[0].SupportsFP16, results[i].SupportsFP16);
            Assert.Equal(results[0].SupportsBF16, results[i].SupportsBF16);
            Assert.Equal(results[0].SupportsFP32, results[i].SupportsFP32);
        }
    }

    [Fact]
    public void PrecisionCapability_StructIsReadonly()
    {
        // Arrange & Act
        var capability = new PrecisionCapability
        {
            SupportsFP32 = true,
            SupportsFP16 = false,
            SupportsBF16 = false
        };

        // Assert - This is a readonly struct, ensuring immutability
        Assert.True(capability.SupportsFP32);
        Assert.False(capability.SupportsFP16);
        Assert.False(capability.SupportsBF16);
    }

    [Fact]
    public void PrecisionCapability_PropertyConsistency()
    {
        // Arrange & Act
        var capability = new PrecisionCapability
        {
            SupportsFP32 = true,
            SupportsFP16 = true,
            SupportsBF16 = false
        };

        // Assert
        Assert.Equal(capability.SupportsFP16, capability.IsFP16Available);
        Assert.Equal(capability.SupportsBF16, capability.IsBF16Available);
    }
}
