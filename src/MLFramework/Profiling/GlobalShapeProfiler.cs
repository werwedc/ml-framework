using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace MLFramework.Profiling
{
    /// <summary>
    /// Global shape profiler with thread-safe operations and persistence
    /// </summary>
    public class GlobalShapeProfiler : IShapeProfiler
    {
        private readonly object _lock = new object();
        private readonly Random _random = new Random();

        /// <summary>
        /// Histogram for each tensor name
        /// </summary>
        public Dictionary<string, ShapeHistogram> TensorHistograms { get; }

        /// <summary>
        /// Maximum number of samples to keep per tensor (default 10000)
        /// </summary>
        public int MaxSamplesPerTensor { get; set; }

        /// <summary>
        /// Directory for persisting profiles
        /// </summary>
        public string? PersistencePath { get; set; }

        public GlobalShapeProfiler(int maxSamplesPerTensor = 10000)
        {
            TensorHistograms = new Dictionary<string, ShapeHistogram>();
            MaxSamplesPerTensor = maxSamplesPerTensor;
        }

        /// <summary>
        /// Record a shape sample with thread safety
        /// </summary>
        public void RecordShape(string tensorName, string opName, int[] shape)
        {
            if (string.IsNullOrEmpty(tensorName))
                throw new ArgumentException("Tensor name cannot be null or empty", nameof(tensorName));

            if (string.IsNullOrEmpty(opName))
                throw new ArgumentException("Operation name cannot be null or empty", nameof(opName));

            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            lock (_lock)
            {
                if (!TensorHistograms.ContainsKey(tensorName))
                {
                    TensorHistograms[tensorName] = new ShapeHistogram();
                }

                var histogram = TensorHistograms[tensorName];

                // Reservoir sampling if we exceed the limit
                if (histogram.TotalSamples >= MaxSamplesPerTensor)
                {
                    // Probability to replace: 1/N
                    double replaceProbability = 1.0 / (histogram.TotalSamples + 1);
                    if (_random.NextDouble() < replaceProbability)
                    {
                        // Find a random sample to replace (simplified: just add the new one)
                        histogram.AddSample(shape);
                    }
                    // Else skip this sample
                }
                else
                {
                    histogram.AddSample(shape);
                }
            }
        }

        /// <summary>
        /// Get the histogram for a specific tensor
        /// </summary>
        public ShapeHistogram? GetHistogram(string tensorName)
        {
            lock (_lock)
            {
                return TensorHistograms.TryGetValue(tensorName, out var histogram) ? histogram : null;
            }
        }

        /// <summary>
        /// Get the most common shapes for a tensor
        /// </summary>
        public List<int[]> GetCommonShapes(string tensorName, int count)
        {
            lock (_lock)
            {
                if (!TensorHistograms.ContainsKey(tensorName))
                    return new List<int[]>();

                return TensorHistograms[tensorName]
                    .GetTopShapes(count)
                    .Select(x => x.shape)
                    .ToList();
            }
        }

        /// <summary>
        /// Get statistical analysis for a tensor
        /// </summary>
        public ShapeStatistics? GetShapeStatistics(string tensorName)
        {
            lock (_lock)
            {
                if (!TensorHistograms.ContainsKey(tensorName))
                    return null;

                var histogram = TensorHistograms[tensorName];
                var statistics = new ShapeStatistics();
                statistics.CalculateFromHistogram(histogram);
                return statistics;
            }
        }

        /// <summary>
        /// Clear profiling data for a specific tensor
        /// </summary>
        public void Clear(string tensorName)
        {
            lock (_lock)
            {
                if (TensorHistograms.ContainsKey(tensorName))
                {
                    TensorHistograms.Remove(tensorName);
                }
            }
        }

        /// <summary>
        /// Clear all profiling data
        /// </summary>
        public void ClearAll()
        {
            lock (_lock)
            {
                TensorHistograms.Clear();
            }
        }

        /// <summary>
        /// Generate a full profiling report
        /// </summary>
        public string GetReport()
        {
            lock (_lock)
            {
                var sb = new System.Text.StringBuilder();
                sb.AppendLine("Global Shape Profiler Report");
                sb.AppendLine("============================");
                sb.AppendLine($"Tensors Profiled: {TensorHistograms.Count}");
                sb.AppendLine($"Max Samples Per Tensor: {MaxSamplesPerTensor}");
                sb.AppendLine();

                foreach (var kvp in TensorHistograms.OrderByDescending(x => x.Value.TotalSamples))
                {
                    sb.AppendLine($"Tensor: {kvp.Key}");
                    sb.AppendLine(kvp.Value.ToReport());
                    sb.AppendLine();
                }

                return sb.ToString();
            }
        }

        /// <summary>
        /// Persist profiles to disk
        /// </summary>
        public void Persist(string? path = null)
        {
            var targetPath = path ?? PersistencePath;
            if (string.IsNullOrEmpty(targetPath))
                throw new InvalidOperationException("No persistence path specified");

            lock (_lock)
            {
                var data = new
                {
                    TensorHistograms = TensorHistograms.ToDictionary(
                        kvp => kvp.Key,
                        kvp => new
                        {
                            BinCounts = kvp.Value.BinCounts,
                            TotalSamples = kvp.Value.TotalSamples,
                            MostCommonShape = kvp.Value.MostCommonShape,
                            MostCommonCount = kvp.Value.MostCommonCount
                        }
                    ),
                    MaxSamplesPerTensor
                };

                Directory.CreateDirectory(Path.GetDirectoryName(targetPath) ?? ".");
                var json = JsonSerializer.Serialize(data, new JsonSerializerOptions
                {
                    WriteIndented = true
                });
                File.WriteAllText(targetPath, json);
            }
        }

        /// <summary>
        /// Load profiles from disk
        /// </summary>
        public void Load(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("Profile file not found", path);

            var json = File.ReadAllText(path);
            var data = JsonSerializer.Deserialize<JsonElement>(json);

            lock (_lock)
            {
                if (data.TryGetProperty("MaxSamplesPerTensor", out var maxSamplesProp))
                {
                    MaxSamplesPerTensor = maxSamplesProp.GetInt32();
                }

                if (data.TryGetProperty("TensorHistograms", out var histogramsProp))
                {
                    foreach (var tensorProp in histogramsProp.EnumerateObject())
                    {
                        var histogram = new ShapeHistogram();
                        var histogramData = tensorProp.Value;

                        if (histogramData.TryGetProperty("BinCounts", out var binCountsProp))
                        {
                            foreach (var binCountProp in binCountsProp.EnumerateObject())
                            {
                                histogram.BinCounts[binCountProp.Name] = binCountProp.Value.GetInt32();
                            }
                        }

                        if (histogramData.TryGetProperty("TotalSamples", out var totalSamplesProp))
                        {
                            histogram.GetType().GetProperty("TotalSamples")?.SetValue(histogram, totalSamplesProp.GetInt32());
                        }

                        if (histogramData.TryGetProperty("MostCommonShape", out var mostCommonShapeProp))
                        {
                            var shape = mostCommonShapeProp.EnumerateArray().Select(x => x.GetInt32()).ToArray();
                            histogram.GetType().GetProperty("MostCommonShape")?.SetValue(histogram, shape);
                        }

                        if (histogramData.TryGetProperty("MostCommonCount", out var mostCommonCountProp))
                        {
                            histogram.GetType().GetProperty("MostCommonCount")?.SetValue(histogram, mostCommonCountProp.GetInt32());
                        }

                        TensorHistograms[tensorProp.Name] = histogram;
                    }
                }
            }
        }
    }
}
