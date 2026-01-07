using RitterFramework.Core.Tensor;
using MLFramework.Quantization.Backends;
using MLFramework.Quantization.Backends.CPUBackend;
using MLFramework.Quantization.DataStructures;
using Xunit;
using System.Diagnostics;

namespace MLFramework.Tests.Quantization.Backends
{
    /// <summary>
    /// Performance comparison tests for different backends.
    /// </summary>
    [Collection("Performance Tests")]
    public class PerformanceComparisonTests
    {
        private readonly CPUBackend _cpuBackend;

        public PerformanceComparisonTests()
        {
            _cpuBackend = new CPUBackend();
        }

        [Fact(Skip = "Performance test - skip in CI")]
        public void CPUBackend_MatMulInt8_PerformanceIsReasonable()
        {
            // Arrange
            var matrixA = new Tensor(new float[1024 * 1024], new int[] { 1024, 1024 }, false, DataType.Int8);
            var matrixB = new Tensor(new float[1024 * 1024], new int[] { 1024, 1024 }, false, DataType.Int8);

            // Warmup
            _cpuBackend.MatMulInt8(matrixA, matrixB);

            // Act
            var stopwatch = Stopwatch.StartNew();
            var result = _cpuBackend.MatMulInt8(matrixA, matrixB);
            stopwatch.Stop();

            // Assert - Should complete in reasonable time (< 1 second)
            Assert.True(stopwatch.ElapsedMilliseconds < 1000,
                $"MatMul took too long: {stopwatch.ElapsedMilliseconds}ms");
        }

        [Fact(Skip = "Performance test - skip in CI")]
        public void CPUBackend_Conv2DInt8_PerformanceIsReasonable()
        {
            // Arrange
            var input = new Tensor(new float[1 * 64 * 224 * 224], new int[] { 1, 64, 224, 224 }, false, DataType.Int8);
            var weights = new Tensor(new float[128 * 64 * 3 * 3], new int[] { 128, 64, 3, 3 }, false, DataType.Int8);
            var stride = new int[] { 1, 1 };
            var padding = new int[] { 1, 1 };
            var dilation = new int[] { 1, 1 };

            // Warmup
            _cpuBackend.Conv2DInt8(input, weights, null, stride, padding, dilation);

            // Act
            var stopwatch = Stopwatch.StartNew();
            var result = _cpuBackend.Conv2DInt8(input, weights, null, stride, padding, dilation);
            stopwatch.Stop();

            // Assert - Should complete in reasonable time (< 5 seconds)
            Assert.True(stopwatch.ElapsedMilliseconds < 5000,
                $"Conv2D took too long: {stopwatch.ElapsedMilliseconds}ms");
        }

        [Fact]
        public void CPUBackend_QuantizePerformance_IsReasonable()
        {
            // Arrange
            var input = new Tensor(new float[1024 * 1024], new int[] { 1024, 1024 });
            var parameters = new QuantizationParameters(
                scale: 1.0f,
                zeroPoint: 0,
                mode: QuantizationMode.PerTensorSymmetric);

            // Warmup
            _cpuBackend.Quantize(input, parameters);

            // Act
            var stopwatch = Stopwatch.StartNew();
            var result = _cpuBackend.Quantize(input, parameters);
            stopwatch.Stop();

            // Assert - Should complete quickly (< 100ms)
            Assert.True(stopwatch.ElapsedMilliseconds < 100,
                $"Quantize took too long: {stopwatch.ElapsedMilliseconds}ms");
        }
    }
}
