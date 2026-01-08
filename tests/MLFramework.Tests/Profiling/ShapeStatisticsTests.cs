using System;
using System.Linq;
using MLFramework.Profiling;
using Xunit;

namespace MLFramework.Tests.Profiling
{
    public class ShapeStatisticsTests
    {
        [Fact]
        public void Constructor_InitializesWithEmptyPercentiles()
        {
            var stats = new ShapeStatistics();

            Assert.NotNull(stats.Percentiles);
            Assert.Empty(stats.Percentiles);
        }

        [Fact]
        public void CalculateFromHistogram_ComputesMeanCorrectly()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10, 20 });
            histogram.AddSample(new[] { 20, 30 });
            histogram.AddSample(new[] { 30, 40 });

            var stats = new ShapeStatistics();
            stats.CalculateFromHistogram(histogram);

            Assert.NotNull(stats.MeanShape);
            Assert.Equal(2, stats.MeanShape.Length);
            Assert.Equal(20.0, stats.MeanShape[0], 1);
            Assert.Equal(30.0, stats.MeanShape[1], 1);
        }

        [Fact]
        public void CalculateFromHistogram_ComputesStdDevCorrectly()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10, 20 });
            histogram.AddSample(new[] { 20, 30 });
            histogram.AddSample(new[] { 30, 40 });

            var stats = new ShapeStatistics();
            stats.CalculateFromHistogram(histogram);

            Assert.NotNull(stats.StdDevShape);
            Assert.Equal(2, stats.StdDevShape.Length);
            Assert.Equal(10.0, stats.StdDevShape[0], 1);
            Assert.Equal(10.0, stats.StdDevShape[1], 1);
        }

        [Fact]
        public void CalculateFromHistogram_ComputesMinAndMaxCorrectly()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10, 20 });
            histogram.AddSample(new[] { 20, 30 });
            histogram.AddSample(new[] { 30, 40 });

            var stats = new ShapeStatistics();
            stats.CalculateFromHistogram(histogram);

            Assert.NotNull(stats.MinShape);
            Assert.NotNull(stats.MaxShape);
            Assert.Equal(2, stats.MinShape.Length);
            Assert.Equal(2, stats.MaxShape.Length);
            Assert.Equal(10, stats.MinShape[0]);
            Assert.Equal(20, stats.MinShape[1]);
            Assert.Equal(30, stats.MaxShape[0]);
            Assert.Equal(40, stats.MaxShape[1]);
        }

        [Fact]
        public void CalculateFromHistogram_ComputesPercentilesCorrectly()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10 });
            histogram.AddSample(new[] { 20 });
            histogram.AddSample(new[] { 30 });
            histogram.AddSample(new[] { 40 });
            histogram.AddSample(new[] { 50 });

            var stats = new ShapeStatistics();
            stats.CalculateFromHistogram(histogram);

            Assert.True(stats.Percentiles.ContainsKey(25));
            Assert.True(stats.Percentiles.ContainsKey(50));
            Assert.True(stats.Percentiles.ContainsKey(75));

            // 50th percentile (median) should be around 30
            Assert.Equal(30, stats.Percentiles[50][0]);
        }

        [Fact]
        public void CalculateFromHistogram_ThrowsOnNullHistogram()
        {
            var stats = new ShapeStatistics();

            Assert.Throws<ArgumentNullException>(() => stats.CalculateFromHistogram(null!));
        }

        [Fact]
        public void CalculateFromHistogram_HandlesEmptyHistogram()
        {
            var histogram = new ShapeHistogram();
            var stats = new ShapeStatistics();

            stats.CalculateFromHistogram(histogram);

            Assert.Null(stats.MeanShape);
            Assert.Null(stats.StdDevShape);
            Assert.Null(stats.MinShape);
            Assert.Null(stats.MaxShape);
        }

        [Fact]
        public void CalculateFromHistogram_HandlesWeightedSamples()
        {
            var histogram = new ShapeHistogram();

            // Add the same shape multiple times
            histogram.AddSample(new[] { 10 });
            histogram.AddSample(new[] { 10 });
            histogram.AddSample(new[] { 20 });
            histogram.AddSample(new[] { 20 });
            histogram.AddSample(new[] { 30 });

            var stats = new ShapeStatistics();
            stats.CalculateFromHistogram(histogram);

            // Mean should be (10*2 + 20*2 + 30*1) / 5 = 90 / 5 = 18
            Assert.Equal(18.0, stats.MeanShape[0], 1);
        }

        [Fact]
        public void CalculateFromHistogram_MultiDimensionalShapes()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 15, 25, 35 });
            histogram.AddSample(new[] { 20, 30, 40 });

            var stats = new ShapeStatistics();
            stats.CalculateFromHistogram(histogram);

            Assert.Equal(3, stats.MeanShape.Length);
            Assert.Equal(15.0, stats.MeanShape[0], 1);
            Assert.Equal(25.0, stats.MeanShape[1], 1);
            Assert.Equal(35.0, stats.MeanShape[2], 1);
        }
    }
}
