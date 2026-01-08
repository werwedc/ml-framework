using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Visualization.GPU
{
    /// <summary>
    /// Main implementation of GPU utilization tracking
    /// </summary>
    public class GPUUtilizationTracker : IGPUTracker, IDisposable
    {
        private readonly CancellationTokenSource _cancellationTokenSource;
        private Task? _samplingTask;
        private readonly ConcurrentDictionary<int, List<GPUUtilizationSample>> _samples;
        private readonly ConcurrentDictionary<int, GPUDeviceInfo> _deviceInfoCache;
        private readonly object _lock = new object();

        public bool IsTracking { get; private set; }
        public bool IsEnabled { get; private set; }

        public int DefaultSamplingIntervalMs { get; set; } = 1000;
        public bool TrackTemperature { get; set; } = true;
        public bool TrackPower { get; set; } = true;

        public GPUUtilizationTracker()
        {
            _cancellationTokenSource = new CancellationTokenSource();
            _samples = new ConcurrentDictionary<int, List<GPUUtilizationSample>>();
            _deviceInfoCache = new ConcurrentDictionary<int, GPUDeviceInfo>();
            IsEnabled = true;
            IsTracking = false;
        }

        #region Device Information

        public List<GPUDeviceInfo> GetAvailableDevices()
        {
            if (!IsEnabled)
                return new List<GPUDeviceInfo>();

            // Cache devices if not already cached
            if (!_deviceInfoCache.Any())
            {
                var devices = QueryGPUDevices();
                foreach (var device in devices)
                {
                    _deviceInfoCache.TryAdd(device.DeviceId, device);
                }
            }

            return _deviceInfoCache.Values.OrderBy(d => d.DeviceId).ToList();
        }

        public GPUDeviceInfo GetDeviceInfo(int deviceId)
        {
            if (!_deviceInfoCache.ContainsKey(deviceId))
            {
                var devices = GetAvailableDevices();
            }

            if (_deviceInfoCache.TryGetValue(deviceId, out var deviceInfo))
                return deviceInfo;

            throw new ArgumentException($"GPU device {deviceId} not found", nameof(deviceId));
        }

        #endregion

        #region Sampling

        public void StartTracking(int deviceId = -1)
        {
            if (!IsEnabled)
                throw new InvalidOperationException("GPU tracker is disabled");

            if (IsTracking)
                throw new InvalidOperationException("Tracking is already in progress");

            lock (_lock)
            {
                if (IsTracking) return;

                IsTracking = true;

                // Initialize samples collection
                var devicesToTrack = deviceId == -1 
                    ? GetAvailableDevices().Select(d => d.DeviceId).ToList()
                    : new List<int> { deviceId };

                foreach (var devId in devicesToTrack)
                {
                    if (!_samples.ContainsKey(devId))
                    {
                        _samples.TryAdd(devId, new List<GPUUtilizationSample>());
                    }
                }

                // Start sampling task
                _samplingTask = Task.Run(() => SamplingLoop(devicesToTrack), _cancellationTokenSource.Token);
            }
        }

        public void StopTracking()
        {
            lock (_lock)
            {
                if (!IsTracking) return;

                _cancellationTokenSource.Cancel();

                try
                {
                    _samplingTask?.Wait(5000); // Wait up to 5 seconds for graceful shutdown
                }
                catch (AggregateException)
                {
                    // Task was canceled - this is expected
                }

                IsTracking = false;
            }
        }

        public void SampleUtilization()
        {
            if (!IsEnabled)
                return;

            var devices = GetAvailableDevices();
            foreach (var device in devices)
            {
                try
                {
                    var sample = SampleDevice(device.DeviceId);
                    if (sample != null)
                    {
                        AddSample(sample);
                    }
                }
                catch (Exception ex)
                {
                    // Log error but continue sampling other devices
                    Console.WriteLine($"Error sampling GPU {device.DeviceId}: {ex.Message}");
                }
            }
        }

        #endregion

        #region Statistics

        public GPUStatistics GetStatistics(int deviceId)
        {
            if (!_samples.TryGetValue(deviceId, out var samples))
                throw new ArgumentException($"No samples available for device {deviceId}", nameof(deviceId));

            if (samples.Count == 0)
                throw new ArgumentException($"No samples collected for device {deviceId}", nameof(deviceId));

            return new GPUStatistics(deviceId, samples);
        }

        public Dictionary<int, GPUStatistics> GetAllStatistics()
        {
            var statistics = new Dictionary<int, GPUStatistics>();
            foreach (var kvp in _samples)
            {
                if (kvp.Value.Count > 0)
                {
                    statistics[kvp.Key] = new GPUStatistics(kvp.Key, kvp.Value);
                }
            }
            return statistics;
        }

        public IEnumerable<GPUUtilizationSample> GetSamples(int deviceId, DateTime start, DateTime end)
        {
            if (!_samples.TryGetValue(deviceId, out var samples))
                return Enumerable.Empty<GPUUtilizationSample>();

            return samples.Where(s => s.Timestamp >= start && s.Timestamp <= end);
        }

        #endregion

        #region Configuration

        public void SetSamplingInterval(TimeSpan interval)
        {
            if (interval.TotalMilliseconds < 100)
                throw new ArgumentException("Sampling interval must be at least 100ms", nameof(interval));

            DefaultSamplingIntervalMs = (int)interval.TotalMilliseconds;
        }

        public void Enable()
        {
            IsEnabled = true;
        }

        public void Disable()
        {
            if (IsTracking)
                StopTracking();

            IsEnabled = false;
        }

        #endregion

        #region Private Methods

        private async Task SamplingLoop(List<int> deviceIds)
        {
            while (!_cancellationTokenSource.Token.IsCancellationRequested)
            {
                try
                {
                    foreach (var deviceId in deviceIds)
                    {
                    var sample = SampleDevice(deviceId);
                    if (sample != null)
                    {
                        AddSample(sample);
                    }
                    }

                    await Task.Delay(DefaultSamplingIntervalMs, _cancellationTokenSource.Token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error in sampling loop: {ex.Message}");
                }
            }
        }

        private GPUUtilizationSample? SampleDevice(int deviceId)
        {
            // This is a stub implementation
            // In a real implementation, this would call vendor-specific APIs
            // For now, we'll use the generic tracker to get mock data
            
            var random = new Random();
            var deviceInfo = GetDeviceInfo(deviceId);
            
            // Generate mock data for demonstration
            float utilization = (float)(random.NextDouble() * 100);
            long usedMemory = (long)(random.NextDouble() * deviceInfo.TotalMemoryBytes);
            long freeMemory = deviceInfo.TotalMemoryBytes - usedMemory;
            
            float temperature = TrackTemperature ? 30f + (float)(random.NextDouble() * 40) : -1f;
            float power = TrackPower ? 50f + (float)(random.NextDouble() * 100) : -1f;
            long fanSpeed = TrackPower ? 1000L + (long)(random.NextDouble() * 3000) : -1L;

            return new GPUUtilizationSample(
                deviceId,
                utilization,
                usedMemory,
                freeMemory,
                deviceInfo.TotalMemoryBytes,
                temperature,
                power,
                fanSpeed);
        }

        private void AddSample(GPUUtilizationSample sample)
        {
            _samples.AddOrUpdate(
                sample.DeviceId,
                new List<GPUUtilizationSample> { sample },
                (key, list) =>
                {
                    lock (list)
                    {
                        list.Add(sample);
                        // Limit samples to prevent memory issues (keep last 10000 samples)
                        if (list.Count > 10000)
                        {
                            list.RemoveAt(0);
                        }
                    }
                    return list;
                });
        }

        private List<GPUDeviceInfo> QueryGPUDevices()
        {
            // This is a stub implementation
            // In a real implementation, this would query vendor-specific APIs
            // For now, return a mock device
            
            return new List<GPUDeviceInfo>
            {
                new GPUDeviceInfo(
                    0,
                    "Mock GPU Device",
                    8L * 1024 * 1024 * 1024, // 8 GB
                    7,
                    5,
                    true)
            };
        }

        #endregion

        #region IDisposable

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                StopTracking();
                _cancellationTokenSource.Dispose();
            }
        }

        #endregion
    }
}
