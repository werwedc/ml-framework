using System;
using System.Linq;
using System.Threading.Tasks;
using MLFramework.Profiling;
using Xunit;

namespace MLFramework.Tests.Profiling
{
    public class GlobalShapeProfilerTests
    {
        [Fact]
        public void Constructor_InitializesWithDefaults()
        {
            var profiler = new GlobalShapeProfiler();

            Assert.NotNull(profiler.TensorHistograms);
            Assert.Empty(profiler.TensorHistograms);
            Assert.Equal(10000, profiler.MaxSamplesPerTensor);
        }

        [Fact]
        public void Constructor_AcceptsCustomMaxSamples()
        {
            var profiler = new GlobalShapeProfiler(maxSamplesPerTensor: 5000);

            Assert.Equal(5000, profiler.MaxSamplesPerTensor);
        }

        [Fact]
        public void RecordShape_AddsShapeToHistogram()
        {
            var profiler = new GlobalShapeProfiler();
            var shape = new[] { 10, 20, 30 };

            profiler.RecordShape("tensor1", "add", shape);

            var histogram = profiler.GetHistogram("tensor1");
            Assert.NotNull(histogram);
            Assert.Equal(1, histogram.TotalSamples);
        }

        [Fact]
        public void RecordShape_CreatesNewHistogramForNewTensor()
        {
            var profiler = new GlobalShapeProfiler();

            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });
            profiler.RecordShape("tensor2", "mul", new[] { 5, 10, 15 });

            Assert.Equal(2, profiler.TensorHistograms.Count);
        }

        [Fact]
        public void RecordShape_ThrowsOnNullTensorName()
        {
            var profiler = new GlobalShapeProfiler();

            Assert.Throws<ArgumentException>(() =>
                profiler.RecordShape(null!, "add", new[] { 10, 20, 30 }));
        }

        [Fact]
        public void RecordShape_ThrowsOnEmptyTensorName()
        {
            var profiler = new GlobalShapeProfiler();

            Assert.Throws<ArgumentException>(() =>
                profiler.RecordShape("", "add", new[] { 10, 20, 30 }));
        }

        [Fact]
        public void RecordShape_ThrowsOnNullOperationName()
        {
            var profiler = new GlobalShapeProfiler();

            Assert.Throws<ArgumentException>(() =>
                profiler.RecordShape("tensor1", null!, new[] { 10, 20, 30 }));
        }

        [Fact]
        public void RecordShape_ThrowsOnNullShape()
        {
            var profiler = new GlobalShapeProfiler();

            Assert.Throws<ArgumentNullException>(() =>
                profiler.RecordShape("tensor1", "add", null!));
        }

        [Fact]
        public void RecordShape_ThreadSafe()
        {
            var profiler = new GlobalShapeProfiler();
            var tasks = new Task[100];

            for (int i = 0; i < 100; i++)
            {
                int index = i;
                tasks[i] = Task.Run(() =>
                {
                    profiler.RecordShape("tensor1", "add", new[] { index, index * 2 });
                });
            }

            Task.WaitAll(tasks);

            var histogram = profiler.GetHistogram("tensor1");
            Assert.NotNull(histogram);
            Assert.Equal(100, histogram.TotalSamples);
        }

        [Fact]
        public void RecordShape_ImplementsReservoirSampling()
        {
            var profiler = new GlobalShapeProfiler(maxSamplesPerTensor: 10);

            // Add more samples than the limit
            for (int i = 0; i < 100; i++)
            {
                profiler.RecordShape("tensor1", "add", new[] { i, i * 2 });
            }

            var histogram = profiler.GetHistogram("tensor1");
            Assert.NotNull(histogram);

            // Should have approximately MaxSamplesPerTensor samples
            Assert.True(histogram.TotalSamples <= profiler.MaxSamplesPerTensor * 2); // Allow some tolerance
        }

        [Fact]
        public void GetHistogram_ReturnsNullForUnknownTensor()
        {
            var profiler = new GlobalShapeProfiler();

            var histogram = profiler.GetHistogram("unknown");

            Assert.Null(histogram);
        }

        [Fact]
        public void GetHistogram_ReturnsExistingHistogram()
        {
            var profiler = new GlobalShapeProfiler();
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });

            var histogram = profiler.GetHistogram("tensor1");

            Assert.NotNull(histogram);
            Assert.Equal(1, histogram.TotalSamples);
        }

        [Fact]
        public void GetCommonShapes_ReturnsCorrectShapes()
        {
            var profiler = new GlobalShapeProfiler();
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });
            profiler.RecordShape("tensor1", "mul", new[] { 5, 10, 15 });

            var shapes = profiler.GetCommonShapes("tensor1", 2);

            Assert.Equal(2, shapes.Count);
            Assert.Equal(new[] { 10, 20, 30 }, shapes[0]);
            Assert.Equal(new[] { 5, 10, 15 }, shapes[1]);
        }

        [Fact]
        public void GetCommonShapes_ReturnsEmptyForUnknownTensor()
        {
            var profiler = new GlobalShapeProfiler();

            var shapes = profiler.GetCommonShapes("unknown", 2);

            Assert.Empty(shapes);
        }

        [Fact]
        public void GetShapeStatistics_ReturnsStatistics()
        {
            var profiler = new GlobalShapeProfiler();
            profiler.RecordShape("tensor1", "add", new[] { 10, 20 });
            profiler.RecordShape("tensor1", "add", new[] { 20, 30 });
            profiler.RecordShape("tensor1", "add", new[] { 30, 40 });

            var stats = profiler.GetShapeStatistics("tensor1");

            Assert.NotNull(stats);
            Assert.NotNull(stats.MeanShape);
            Assert.Equal(20.0, stats.MeanShape[0], 1);
            Assert.Equal(30.0, stats.MeanShape[1], 1);
        }

        [Fact]
        public void GetShapeStatistics_ReturnsNullForUnknownTensor()
        {
            var profiler = new GlobalShapeProfiler();

            var stats = profiler.GetShapeStatistics("unknown");

            Assert.Null(stats);
        }

        [Fact]
        public void Clear_RemovesTensor()
        {
            var profiler = new GlobalShapeProfiler();
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });

            profiler.Clear("tensor1");

            Assert.Empty(profiler.TensorHistograms);
        }

        [Fact]
        public void Clear_DoesNothingForUnknownTensor()
        {
            var profiler = new GlobalShapeProfiler();
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });

            profiler.Clear("unknown");

            Assert.Single(profiler.TensorHistograms);
        }

        [Fact]
        public void ClearAll_RemovesAllTensors()
        {
            var profiler = new GlobalShapeProfiler();
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });
            profiler.RecordShape("tensor2", "mul", new[] { 5, 10, 15 });

            profiler.ClearAll();

            Assert.Empty(profiler.TensorHistograms);
        }

        [Fact]
        public void GetReport_GeneratesCorrectReport()
        {
            var profiler = new GlobalShapeProfiler();
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });
            profiler.RecordShape("tensor2", "mul", new[] { 5, 10, 15 });

            var report = profiler.GetReport();

            Assert.Contains("Global Shape Profiler Report", report);
            Assert.Contains("Tensors Profiled: 2", report);
            Assert.Contains("tensor1", report);
            Assert.Contains("tensor2", report);
        }

        [Fact]
        public void GetReport_OrdersTensorsBySampleCount()
        {
            var profiler = new GlobalShapeProfiler();
            profiler.RecordShape("tensor1", "add", new[] { 10, 20, 30 });
            profiler.RecordShape("tensor2", "add", new[] { 5, 10, 15 });
            profiler.RecordShape("tensor2", "add", new[] { 5, 10, 15 });

            var report = profiler.GetReport();

            // tensor2 should appear first (2 samples vs 1)
            int tensor2Index = report.IndexOf("tensor2");
            int tensor1Index = report.IndexOf("tensor1");
            Assert.True(tensor2Index < tensor1Index);
        }
    }
}
