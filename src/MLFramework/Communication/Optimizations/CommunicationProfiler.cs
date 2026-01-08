using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Diagnostics;

namespace MLFramework.Communication.Optimizations
{
    /// <summary>
    /// Profiler for communication operations
    /// </summary>
    public class CommunicationProfiler : IDisposable
    {
        private readonly List<CommunicationProfile> _profiles;
        private readonly object _lock;
        private readonly bool _enabled;
        private bool _disposed;

        public IReadOnlyList<CommunicationProfile> Profiles
        {
            get
            {
                lock (_lock)
                {
                    return _profiles.ToList();
                }
            }
        }

        public CommunicationProfiler(bool enabled = true)
        {
            _enabled = enabled;
            _profiles = new List<CommunicationProfile>();
            _lock = new object();
        }

        /// <summary>
        /// Profile a communication operation
        /// </summary>
        public T Profile<T>(string operation, long dataSizeBytes, Func<T> func, int numRanks = 0, string algorithm = "")
        {
            if (!_enabled)
            {
                return func();
            }

            var stopwatch = Stopwatch.StartNew();
            var result = func();
            stopwatch.Stop();

            var profile = new CommunicationProfile
            {
                Operation = operation,
                DataSizeBytes = dataSizeBytes,
                Duration = stopwatch.Elapsed,
                BandwidthMBps = CalculateBandwidth(dataSizeBytes, stopwatch.Elapsed),
                NumRanks = numRanks,
                Algorithm = algorithm,
                Timestamp = DateTime.Now
            };

            lock (_lock)
            {
                _profiles.Add(profile);
            }

            return result;
        }

        /// <summary>
        /// Profile async operation
        /// </summary>
        public async Task<T> ProfileAsync<T>(string operation, long dataSizeBytes, Func<Task<T>> func, int numRanks = 0, string algorithm = "")
        {
            if (!_enabled)
            {
                return await func();
            }

            var stopwatch = Stopwatch.StartNew();
            var result = await func();
            stopwatch.Stop();

            var profile = new CommunicationProfile
            {
                Operation = operation,
                DataSizeBytes = dataSizeBytes,
                Duration = stopwatch.Elapsed,
                BandwidthMBps = CalculateBandwidth(dataSizeBytes, stopwatch.Elapsed),
                NumRanks = numRanks,
                Algorithm = algorithm,
                Timestamp = DateTime.Now
            };

            lock (_lock)
            {
                _profiles.Add(profile);
            }

            return result;
        }

        private double CalculateBandwidth(long dataSizeBytes, TimeSpan duration)
        {
            if (duration.TotalSeconds == 0)
                return 0;

            return (dataSizeBytes / 1024.0 / 1024.0) / duration.TotalSeconds;
        }

        /// <summary>
        /// Clear all profiles
        /// </summary>
        public void Clear()
        {
            lock (_lock)
            {
                _profiles.Clear();
            }
        }

        /// <summary>
        /// Get statistics
        /// </summary>
        public CommunicationProfileStatistics GetStatistics()
        {
            lock (_lock)
            {
                if (_profiles.Count == 0)
                {
                    return new CommunicationProfileStatistics();
                }

                return new CommunicationProfileStatistics
                {
                    TotalOperations = _profiles.Count,
                    TotalDataTransferred = _profiles.Sum(p => p.DataSizeBytes),
                    TotalTime = _profiles.Sum(p => p.Duration.TotalMilliseconds),
                    AverageBandwidth = _profiles.Average(p => p.BandwidthMBps),
                    MinBandwidth = _profiles.Min(p => p.BandwidthMBps),
                    MaxBandwidth = _profiles.Max(p => p.BandwidthMBps)
                };
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                lock (_lock)
                {
                    _profiles.Clear();
                }
                _disposed = true;
            }
        }
    }
}
