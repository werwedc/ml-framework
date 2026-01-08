using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using MLFramework.Visualization.GPU;

namespace MLFramework.Visualization.Tests.GPU
{
    public class GPUStatisticsTests
    {
        [Fact]
        public void GPUStatistics_Constructor_ValidatesParameters()
        {
            var samples = CreateTestSamples(10);

            // Valid construction
            var stats = new GPUStatistics(0, samples);
            Assert.Equal(0, stats.DeviceId);
            Assert.NotNull(stats.Samples);
            Assert.Equal(10, stats.Samples.Count);

            // Invalid device ID
            Assert.Throws<ArgumentException>(() => new GPUStatistics(-1, samples));

            // Empty samples
            Assert.Throws<ArgumentException>(() => new GPUStatistics(0, new List<GPUUtilizationSample>()));
            
            // Null samples
            Assert.Throws<ArgumentNullException>(() => new GPUStatistics(0, null!));
        }

        [Fact]
        public void GPUStatistics_CalculatesUtilizationStatistics()
        {
            var samples = CreateTestSamples(100);
            var stats = new GPUStatistics(0, samples);

            Assert.InRange(stats.AverageUtilizationPercent, 0, 100);
            Assert.InRange(stats.MaxUtilizationPercent, 0, 100);
            Assert.InRange(stats.MinUtilizationPercent, 0, 100);
            Assert.True(stats.MaxUtilizationPercent >= stats.MinUtilizationPercent);
        }

        [Fact]
        public void GPUStatistics_CalculatesMemoryStatistics()
        {
            var samples = CreateTestSamples(100);
            var stats = new GPUStatistics(0, samples);

            Assert.True(stats.PeakUsedMemoryBytes > 0);
            Assert.InRange(stats.AverageMemoryUsagePercent, 0, 100);
            Assert.NotNull(stats.TotalAllocations);
            Assert.NotNull(stats.TotalDeallocations);
        }

        [Fact]
        public void GPUStatistics_CalculatesIdleTime()
        {
            // Create samples with specific idle pattern
            var samples = new List<GPUUtilizationSample>();
            var baseTime = DateTime.UtcNow;

            for (int i = 0; i < 50; i++)
            {
                // 50% idle samples (utilization < 5%)
                var utilization = i % 2 == 0 ? 2f : 50f;
                samples.Add(new GPUUtilizationSample(
                    0,
                    utilization,
                    4L * 1024 * 1024 * 1024,
                    4L * 1024 * 1024 * 1024,
                    8L * 1024 * 1024 * 1024,
                    -1f,
                    -1f,
                    -1,
                    baseTime.AddSeconds(i)));
            }

            var stats = new GPUStatistics(0, samples, 5f); // 5% idle threshold

            Assert.True(stats.TotalIdleTime > TimeSpan.Zero);
            Assert.True(stats.IdleTimePercent > 0);
        }

        [Fact]
        public void GPUStatistics_Temperature_WhenAvailable()
        {
            var samples = CreateTestSamplesWithTemperature(100);
            var stats = new GPUStatistics(0, samples);

            Assert.True(stats.HasTemperature);
            Assert.InRange(stats.AverageTemperatureCelsius, 0, 100);
            Assert.InRange(stats.MaxTemperatureCelsius, 0, 100);
            Assert.True(stats.MaxTemperatureCelsius >= stats.AverageTemperatureCelsius);
        }

        [Fact]
        public void GPUStatistics_Temperature_WhenNotAvailable()
        {
            var samples = CreateTestSamples(100); // No temperature data
            var stats = new GPUStatistics(0, samples);

            Assert.False(stats.HasTemperature);
            Assert.Equal(-1f, stats.AverageTemperatureCelsius);
            Assert.Equal(-1f, stats.MaxTemperatureCelsius);
        }

        [Fact]
        public void GPUStatistics_Power_WhenAvailable()
        {
            var samples = CreateTestSamplesWithPower(100);
            var stats = new GPUStatistics(0, samples);

            Assert.True(stats.HasPower);
            Assert.InRange(stats.AveragePowerUsageWatts, 0, 1000);
            Assert.InRange(stats.PeakPowerUsageWatts, 0, 1000);
            Assert.True(stats.PeakPowerUsageWatts >= stats.AveragePowerUsageWatts);
        }

        [Fact]
        public void GPUStatistics_Power_WhenNotAvailable()
        {
            var samples = CreateTestSamples(100); // No power data
            var stats = new GPUStatistics(0, samples);

            Assert.False(stats.HasPower);
            Assert.Equal(-1f, stats.AveragePowerUsageWatts);
            Assert.Equal(-1f, stats.PeakPowerUsageWatts);
        }

        [Fact]
        public void GPUStatistics_SamplesArePreserved()
        {
            var samples = CreateTestSamples(10);
            var stats = new GPUStatistics(0, samples);

            Assert.Equal(10, stats.Samples.Count);
            for (int i = 0; i < samples.Count; i++)
            {
                Assert.Equal(samples[i].DeviceId, stats.Samples[i].DeviceId);
                Assert.Equal(samples[i].UtilizationPercent, stats.Samples[i].UtilizationPercent);
            }
        }

        [Fact]
        public void GPUStatistics_SingleSample_HandlesEdgeCase()
        {
            var samples = CreateTestSamples(1);
            var stats = new GPUStatistics(0, samples);

            Assert.Equal(samples[0].UtilizationPercent, stats.AverageUtilizationPercent);
            Assert.Equal(samples[0].UtilizationPercent, stats.MaxUtilizationPercent);
            Assert.Equal(samples[0].UtilizationPercent, stats.MinUtilizationPercent);
        }

        [Fact]
        public void GPUStatistics_HighUtilization_NoIdleTime()
        {
            // Create samples with high utilization (no idle time)
            var samples = new List<GPUUtilizationSample>();
            var baseTime = DateTime.UtcNow;

            for (int i = 0; i < 10; i++)
            {
                samples.Add(new GPUUtilizationSample(
                    0,
                    90f, // High utilization
                    4L * 1024 * 1024 * 1024,
                    4L * 1024 * 1024 * 1024,
                    8L * 1024 * 1024 * 1024,
                    -1f,
                    -1f,
                    -1,
                    baseTime.AddSeconds(i)));
            }

            var stats = new GPUStatistics(0, samples, 5f);

            // With 5% idle threshold and 90% utilization, idle time should be minimal
            Assert.True(stats.IdleTimePercent < 10);
        }

        [Fact]
        public void GPUStatistics_VariableUtilization_DetectsRange()
        {
            var samples = new List<GPUUtilizationSample>();
            var baseTime = DateTime.UtcNow;

            // Create samples with varying utilization
            for (int i = 0; i < 10; i++)
            {
                float utilization = i * 10f; // 0%, 10%, 20%, ..., 90%
                samples.Add(new GPUUtilizationSample(
                    0,
                    utilization,
                    4L * 1024 * 1024 * 1024,
                    4L * 1024 * 1024 * 1024,
                    8L * 1024 * 1024 * 1024,
                    -1f,
                    -1f,
                    -1,
                    baseTime.AddSeconds(i)));
            }

            var stats = new GPUStatistics(0, samples);

            Assert.Equal(0f, stats.MinUtilizationPercent);
            Assert.Equal(90f, stats.MaxUtilizationPercent);
            Assert.InRange(stats.AverageUtilizationPercent, 0, 90);
        }

        private List<GPUUtilizationSample> CreateTestSamples(int count)
        {
            var samples = new List<GPUUtilizationSample>();
            var baseTime = DateTime.UtcNow;
            var random = new Random(42); // Fixed seed for reproducibility

            for (int i = 0; i < count; i++)
            {
                float utilization = (float)(random.NextDouble() * 100);
                long usedMemory = (long)(random.NextDouble() * 8L * 1024 * 1024 * 1024);
                long totalMemory = 8L * 1024 * 1024 * 1024;

                samples.Add(new GPUUtilizationSample(
                    0,
                    utilization,
                    usedMemory,
                    totalMemory - usedMemory,
                    totalMemory,
                    -1f, // No temperature
                    -1f, // No power
                    -1,  // No fan speed
                    baseTime.AddMilliseconds(i)));
            }

            return samples;
        }

        private List<GPUUtilizationSample> CreateTestSamplesWithTemperature(int count)
        {
            var samples = CreateTestSamples(count);
            var random = new Random(42);

            // Add temperature to samples
            for (int i = 0; i < samples.Count; i++)
            {
                // Create a new sample with temperature
                var original = samples[i];
                float temperature = 30f + (float)(random.NextDouble() * 40);

                samples[i] = new GPUUtilizationSample(
                    original.DeviceId,
                    original.UtilizationPercent,
                    original.UsedMemoryBytes,
                    original.FreeMemoryBytes,
                    original.TotalMemoryBytes,
                    temperature,
                    original.PowerUsageWatts,
                    original.FanSpeedRPM,
                    original.Timestamp);
            }

            return samples;
        }

        private List<GPUUtilizationSample> CreateTestSamplesWithPower(int count)
        {
            var samples = CreateTestSamples(count);
            var random = new Random(42);

            // Add power to samples
            for (int i = 0; i < samples.Count; i++)
            {
                // Create a new sample with power
                var original = samples[i];
                float power = 50f + (float)(random.NextDouble() * 100);

                samples[i] = new GPUUtilizationSample(
                    original.DeviceId,
                    original.UtilizationPercent,
                    original.UsedMemoryBytes,
                    original.FreeMemoryBytes,
                    original.TotalMemoryBytes,
                    original.TemperatureCelsius,
                    power,
                    original.FanSpeedRPM,
                    original.Timestamp);
            }

            return samples;
        }
    }
}
