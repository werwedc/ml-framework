using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Visualization.GPU
{
    /// <summary>
    /// Aggregated statistics for GPU utilization
    /// </summary>
    public class GPUStatistics
    {
        public int DeviceId { get; }

        // Utilization
        public float AverageUtilizationPercent { get; private set; }
        public float MaxUtilizationPercent { get; private set; }
        public float MinUtilizationPercent { get; private set; }
        public TimeSpan TotalIdleTime { get; private set; }
        public float IdleTimePercent { get; private set; }

        // Memory
        public long PeakUsedMemoryBytes { get; private set; }
        public float AverageMemoryUsagePercent { get; private set; }
        public long TotalAllocations { get; private set; }
        public long TotalDeallocations { get; private set; }

        // Temperature and power
        public float AverageTemperatureCelsius { get; private set; }
        public float MaxTemperatureCelsius { get; private set; }
        public float AveragePowerUsageWatts { get; private set; }
        public float PeakPowerUsageWatts { get; private set; }

        // Samples
        public List<GPUUtilizationSample> Samples { get; }

        public GPUStatistics(
            int deviceId,
            IEnumerable<GPUUtilizationSample> samples,
            float idleThresholdPercent = 5f)
        {
            if (deviceId < 0)
                throw new ArgumentException("Device ID must be non-negative", nameof(deviceId));
            
            if (samples == null)
                throw new ArgumentNullException(nameof(samples));

            var sampleList = samples.ToList();
            if (!sampleList.Any())
                throw new ArgumentException("At least one sample is required", nameof(samples));

            DeviceId = deviceId;
            Samples = new List<GPUUtilizationSample>(sampleList);
            CalculateStatistics(sampleList, idleThresholdPercent);
        }

        private void CalculateStatistics(List<GPUUtilizationSample> samples, float idleThresholdPercent)
        {
            // Utilization statistics
            AverageUtilizationPercent = samples.Average(s => s.UtilizationPercent);
            MaxUtilizationPercent = samples.Max(s => s.UtilizationPercent);
            MinUtilizationPercent = samples.Min(s => s.UtilizationPercent);

            // Idle time calculation
            var idleSamples = samples.Where(s => s.UtilizationPercent < idleThresholdPercent).ToList();
            if (idleSamples.Count > 1)
            {
                double totalIdleSeconds = 0;
                for (int i = 1; i < idleSamples.Count; i++)
                {
                    var prevSample = idleSamples[i - 1];
                    var currentSample = idleSamples[i];
                    totalIdleSeconds += (currentSample.Timestamp - prevSample.Timestamp).TotalSeconds;
                }
                TotalIdleTime = TimeSpan.FromSeconds(totalIdleSeconds);
            }
            else
            {
                TotalIdleTime = TimeSpan.Zero;
            }

            // Total time span
            var totalTimespan = samples.Last().Timestamp - samples.First().Timestamp;
            IdleTimePercent = totalTimespan.TotalSeconds > 0 
                ? (float)(TotalIdleTime.TotalSeconds / totalTimespan.TotalSeconds * 100)
                : 0f;

            // Memory statistics
            PeakUsedMemoryBytes = samples.Max(s => s.UsedMemoryBytes);
            AverageMemoryUsagePercent = samples.Average(s => s.MemoryUsagePercent);

            // For allocations/deallocations, we estimate from memory changes
            // This is a simplified calculation - in a real system, you'd track this directly
            TotalAllocations = EstimateAllocations(samples);
            TotalDeallocations = EstimateDeallocations(samples);

            // Temperature statistics (only consider samples with temperature data)
            var tempSamples = samples.Where(s => s.HasTemperature).ToList();
            if (tempSamples.Any())
            {
                AverageTemperatureCelsius = tempSamples.Average(s => s.TemperatureCelsius);
                MaxTemperatureCelsius = tempSamples.Max(s => s.TemperatureCelsius);
            }
            else
            {
                AverageTemperatureCelsius = -1f;
                MaxTemperatureCelsius = -1f;
            }

            // Power statistics (only consider samples with power data)
            var powerSamples = samples.Where(s => s.HasPower).ToList();
            if (powerSamples.Any())
            {
                AveragePowerUsageWatts = powerSamples.Average(s => s.PowerUsageWatts);
                PeakPowerUsageWatts = powerSamples.Max(s => s.PowerUsageWatts);
            }
            else
            {
                AveragePowerUsageWatts = -1f;
                PeakPowerUsageWatts = -1f;
            }
        }

        private long EstimateAllocations(List<GPUUtilizationSample> samples)
        {
            // Count increases in memory usage as allocations
            long allocations = 0;
            for (int i = 1; i < samples.Count; i++)
            {
                if (samples[i].UsedMemoryBytes > samples[i - 1].UsedMemoryBytes)
                {
                    allocations++;
                }
            }
            return allocations;
        }

        private long EstimateDeallocations(List<GPUUtilizationSample> samples)
        {
            // Count decreases in memory usage as deallocations
            long deallocations = 0;
            for (int i = 1; i < samples.Count; i++)
            {
                if (samples[i].UsedMemoryBytes < samples[i - 1].UsedMemoryBytes)
                {
                    deallocations++;
                }
            }
            return deallocations;
        }

        /// <summary>
        /// Checks if temperature data is available
        /// </summary>
        public bool HasTemperature => AverageTemperatureCelsius >= 0;

        /// <summary>
        /// Checks if power data is available
        /// </summary>
        public bool HasPower => AveragePowerUsageWatts >= 0;

        /// <summary>
        /// Returns a summary string of the statistics
        /// </summary>
        public override string ToString()
        {
            return $"GPU {DeviceId} Statistics:\n" +
                   $"  Utilization: Avg {AverageUtilizationPercent:F1}% | Max {MaxUtilizationPercent:F1}% | Min {MinUtilizationPercent:F1}%\n" +
                   $"  Idle Time: {TotalIdleTime.TotalMinutes:F1} min ({IdleTimePercent:F1}%)\n" +
                   $"  Memory: Peak {PeakUsedMemoryBytes / (1024 * 1024)} MB | Avg {AverageMemoryUsagePercent:F1}%\n" +
                   (HasTemperature ? $"  Temperature: Avg {AverageTemperatureCelsius:F1}°C | Max {MaxTemperatureCelsius:F1}°C\n" : "") +
                   (HasPower ? $"  Power: Avg {AveragePowerUsageWatts:F1}W | Peak {PeakPowerUsageWatts:F1}W\n" : "") +
                   $"  Total Samples: {Samples.Count}";
        }
    }
}
