using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using MLFramework.Visualization.GPU;

namespace MLFramework.Visualization.Tests.GPU
{
    public class GPUUtilizationTrackerTests
    {
        [Fact]
        public void GPUDeviceInfo_Constructor_ValidatesParameters()
        {
            // Valid construction
            var deviceInfo = new GPUDeviceInfo(0, "Test GPU", 8L * 1024 * 1024 * 1024, 7, 5, true);
            Assert.Equal(0, deviceInfo.DeviceId);
            Assert.Equal("Test GPU", deviceInfo.Name);
            Assert.Equal(8L * 1024 * 1024 * 1024, deviceInfo.TotalMemoryBytes);
            Assert.Equal(7, deviceInfo.ComputeCapabilityMajor);
            Assert.Equal(5, deviceInfo.ComputeCapabilityMinor);
            Assert.True(deviceInfo.IsAvailable);

            // Invalid device ID
            Assert.Throws<ArgumentException>(() => new GPUDeviceInfo(-1, "Test", 1024, 1, 0, true));

            // Empty name
            Assert.Throws<ArgumentException>(() => new GPUDeviceInfo(0, "", 1024, 1, 0, true));
            Assert.Throws<ArgumentException>(() => new GPUDeviceInfo(0, null!, 1024, 1, 0, true));

            // Invalid memory
            Assert.Throws<ArgumentException>(() => new GPUDeviceInfo(0, "Test", 0, 1, 0, true));
            Assert.Throws<ArgumentException>(() => new GPUDeviceInfo(0, "Test", -1, 1, 0, true));
        }

        [Fact]
        public void GPUUtilizationSample_Constructor_ValidatesParameters()
        {
            var timestamp = DateTime.UtcNow;

            // Valid construction
            var sample = new GPUUtilizationSample(
                0,
                50f,
                4L * 1024 * 1024 * 1024,
                4L * 1024 * 1024 * 1024,
                8L * 1024 * 1024 * 1024,
                60f,
                100f,
                2000,
                timestamp);
            
            Assert.Equal(0, sample.DeviceId);
            Assert.Equal(50f, sample.UtilizationPercent);
            Assert.Equal(4L * 1024 * 1024 * 1024, sample.UsedMemoryBytes);
            Assert.Equal(8L * 1024 * 1024 * 1024, sample.TotalMemoryBytes);
            Assert.Equal(60f, sample.TemperatureCelsius);
            Assert.Equal(100f, sample.PowerUsageWatts);
            Assert.Equal(2000, sample.FanSpeedRPM);

            // Invalid device ID
            Assert.Throws<ArgumentException>(() => new GPUUtilizationSample(-1, 50f, 1024, 1024, 2048));

            // Invalid utilization
            Assert.Throws<ArgumentException>(() => new GPUUtilizationSample(0, -1f, 1024, 1024, 2048));
            Assert.Throws<ArgumentException>(() => new GPUUtilizationSample(0, 101f, 1024, 1024, 2048));

            // Invalid memory
            Assert.Throws<ArgumentException>(() => new GPUUtilizationSample(0, 50f, -1, 1024, 2048));
            Assert.Throws<ArgumentException>(() => new GPUUtilizationSample(0, 50f, 1024, -1, 2048));
            Assert.Throws<ArgumentException>(() => new GPUUtilizationSample(0, 50f, 0, 0, 0));

            // Memory exceeds total
            Assert.Throws<ArgumentException>(() => new GPUUtilizationSample(0, 50f, 2048, 2048, 2048));
        }

        [Fact]
        public void GPUUtilizationSample_CalculatedProperties()
        {
            var sample = new GPUUtilizationSample(
                0,
                50f,
                4L * 1024 * 1024 * 1024,
                4L * 1024 * 1024 * 1024,
                8L * 1024 * 1024 * 1024);

            Assert.Equal(50f, sample.MemoryUsagePercent);
            Assert.False(sample.HasTemperature); // Default -1 indicates no temperature
            Assert.False(sample.HasPower);
            Assert.False(sample.HasFanSpeed);
        }

        [Fact]
        public void GPUTracker_GetAvailableDevices_ReturnsDevices()
        {
            var tracker = new GPUUtilizationTracker();
            var devices = tracker.GetAvailableDevices();

            Assert.NotNull(devices);
            Assert.NotEmpty(devices);
            Assert.True(devices.All(d => d.DeviceId >= 0));
        }

        [Fact]
        public void GPUTracker_StartStopTracking_WorksCorrectly()
        {
            var tracker = new GPUUtilizationTracker();
            
            Assert.False(tracker.IsTracking);

            tracker.StartTracking();
            Assert.True(tracker.IsTracking);

            tracker.StopTracking();
            Assert.False(tracker.IsTracking);
        }

        [Fact]
        public void GPUTracker_SampleUtilization_RecordsSamples()
        {
            var tracker = new GPUUtilizationTracker();
            
            tracker.SampleUtilization();
            
            var devices = tracker.GetAvailableDevices();
            foreach (var device in devices)
            {
                var stats = tracker.GetStatistics(device.DeviceId);
                Assert.NotNull(stats);
                Assert.Equal(device.DeviceId, stats.DeviceId);
                Assert.NotEmpty(stats.Samples);
            }
        }

        [Fact]
        public void GPUTracker_Disable_PreventsOperations()
        {
            var tracker = new GPUUtilizationTracker();
            
            tracker.Disable();
            Assert.False(tracker.IsEnabled);

            Assert.Throws<InvalidOperationException>(() => tracker.StartTracking());

            // Re-enable
            tracker.Enable();
            Assert.True(tracker.IsEnabled);
        }

        [Fact]
        public void GPUTracker_SetSamplingInterval_ValidatesInterval()
        {
            var tracker = new GPUUtilizationTracker();
            
            // Valid interval
            tracker.SetSamplingInterval(TimeSpan.FromMilliseconds(1000));

            // Invalid interval
            Assert.Throws<ArgumentException>(() => tracker.SetSamplingInterval(TimeSpan.FromMilliseconds(50)));
        }

        [Fact]
        public async Task GPUTracker_AutoSampling_CollectsSamples()
        {
            var tracker = new GPUUtilizationTracker();
            
            tracker.StartTracking();
            tracker.SetSamplingInterval(TimeSpan.FromMilliseconds(100));

            // Wait for several samples to be collected
            await Task.Delay(500);

            tracker.StopTracking();

            var devices = tracker.GetAvailableDevices();
            foreach (var device in devices)
            {
                var stats = tracker.GetStatistics(device.DeviceId);
                Assert.True(stats.Samples.Count >= 2); // At least 2 samples should be collected
            }
        }

        [Fact]
        public void GPUTracker_GetSamples_ByTimeRange()
        {
            var tracker = new GPUUtilizationTracker();
            
            var startTime = DateTime.UtcNow;
            tracker.SampleUtilization();
            var endTime = DateTime.UtcNow;

            var devices = tracker.GetAvailableDevices();
            foreach (var device in devices)
            {
                var samples = tracker.GetSamples(device.DeviceId, startTime, endTime);
                Assert.NotNull(samples);
                Assert.NotEmpty(samples);
                Assert.True(samples.All(s => s.Timestamp >= startTime && s.Timestamp <= endTime));
            }
        }

        [Fact]
        public void GPUTracker_Dispose_CleansUpResources()
        {
            var tracker = new GPUUtilizationTracker();
            tracker.StartTracking();

            tracker.Dispose();

            Assert.False(tracker.IsTracking);
        }

        [Fact]
        public void GPUTracker_VerifySampleCollection()
        {
            var tracker = new GPUUtilizationTracker();

            tracker.SampleUtilization();

            // Verify that samples were collected
            var devices = tracker.GetAvailableDevices();
            Assert.NotEmpty(devices);
        }
    }
}
