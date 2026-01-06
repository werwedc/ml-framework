using MLFramework.Amp;
using Xunit;

namespace MLFramework.Tests.Amp.Kernel
{
    public class KernelPerformanceStatsTests
    {
        [Fact]
        public void Constructor_SetsAllProperties()
        {
            // Arrange & Act
            var stats = new KernelPerformanceStats(
                "conv2d",
                KernelDtype.Float16,
                10.5f,
                8.0f,
                15.0f,
                100);

            // Assert
            Assert.Equal("conv2d", stats.OperationName);
            Assert.Equal(KernelDtype.Float16, stats.Dtype);
            Assert.Equal(10.5f, stats.AverageExecutionTime);
            Assert.Equal(8.0f, stats.MinExecutionTime);
            Assert.Equal(15.0f, stats.MaxExecutionTime);
            Assert.Equal(100, stats.ExecutionCount);
        }

        [Fact]
        public void ToString_ReturnsFormattedString()
        {
            // Arrange
            var stats = new KernelPerformanceStats(
                "conv2d",
                KernelDtype.Float16,
                10.5f,
                8.0f,
                15.0f,
                100);

            // Act
            var result = stats.ToString();

            // Assert
            Assert.Contains("Operation=conv2d", result);
            Assert.Contains("Dtype=Float16", result);
            Assert.Contains("AvgTime=10.500ms", result);
            Assert.Contains("MinTime=8.000ms", result);
            Assert.Contains("MaxTime=15.000ms", result);
            Assert.Contains("Executions=100", result);
        }

        [Fact]
        public void ToString_WithDifferentPrecision_FormatsCorrectly()
        {
            // Arrange
            var stats = new KernelPerformanceStats(
                "matmul",
                KernelDtype.BFloat16,
                5.12345f,
                5.0f,
                5.5f,
                1);

            // Act
            var result = stats.ToString();

            // Assert
            Assert.Contains("AvgTime=5.123ms", result);
        }

        [Fact]
        public void Constructor_SingleExecution_SameMinMaxAvg()
        {
            // Arrange & Act
            var stats = new KernelPerformanceStats(
                "relu",
                KernelDtype.Float32,
                7.5f,
                7.5f,
                7.5f,
                1);

            // Assert
            Assert.Equal(7.5f, stats.AverageExecutionTime);
            Assert.Equal(7.5f, stats.MinExecutionTime);
            Assert.Equal(7.5f, stats.MaxExecutionTime);
            Assert.Equal(1, stats.ExecutionCount);
        }
    }
}
