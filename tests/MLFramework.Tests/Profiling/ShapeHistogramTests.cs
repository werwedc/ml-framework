using System;
using System.Linq;
using MLFramework.Profiling;
using Xunit;

namespace MLFramework.Tests.Profiling
{
    public class ShapeHistogramTests
    {
        [Fact]
        public void Constructor_InitializesEmptyHistogram()
        {
            var histogram = new ShapeHistogram();

            Assert.NotNull(histogram.BinCounts);
            Assert.Equal(0, histogram.TotalSamples);
            Assert.Equal(0, histogram.UniqueShapes);
            Assert.Null(histogram.MostCommonShape);
            Assert.Equal(0, histogram.MostCommonCount);
        }

        [Fact]
        public void AddSample_AddsShapeToHistogram()
        {
            var histogram = new ShapeHistogram();
            var shape = new[] { 10, 20, 30 };

            histogram.AddSample(shape);

            Assert.Equal(1, histogram.TotalSamples);
            Assert.Equal(1, histogram.UniqueShapes);
            Assert.NotNull(histogram.MostCommonShape);
            Assert.Equal(shape, histogram.MostCommonShape);
            Assert.Equal(1, histogram.MostCommonCount);
        }

        [Fact]
        public void AddSample_IncrementsCountForExistingShape()
        {
            var histogram = new ShapeHistogram();
            var shape = new[] { 10, 20, 30 };

            histogram.AddSample(shape);
            histogram.AddSample(shape);

            Assert.Equal(2, histogram.TotalSamples);
            Assert.Equal(1, histogram.UniqueShapes);
            Assert.Equal(2, histogram.MostCommonCount);
        }

        [Fact]
        public void AddSample_TracksMultipleShapes()
        {
            var histogram = new ShapeHistogram();
            var shape1 = new[] { 10, 20, 30 };
            var shape2 = new[] { 5, 10, 15 };

            histogram.AddSample(shape1);
            histogram.AddSample(shape2);
            histogram.AddSample(shape1);

            Assert.Equal(3, histogram.TotalSamples);
            Assert.Equal(2, histogram.UniqueShapes);
            Assert.Equal(2, histogram.MostCommonCount);
        }

        [Fact]
        public void AddSample_ThrowsOnNullShape()
        {
            var histogram = new ShapeHistogram();

            Assert.Throws<ArgumentNullException>(() => histogram.AddSample(null!));
        }

        [Fact]
        public void GetFrequency_ReturnsCorrectValue()
        {
            var histogram = new ShapeHistogram();
            var shape1 = new[] { 10, 20, 30 };
            var shape2 = new[] { 5, 10, 15 };

            histogram.AddSample(shape1);
            histogram.AddSample(shape1);
            histogram.AddSample(shape2);

            var freq1 = histogram.GetFrequency(shape1);
            var freq2 = histogram.GetFrequency(shape2);

            Assert.Equal(2.0 / 3.0, freq1, 3);
            Assert.Equal(1.0 / 3.0, freq2, 3);
        }

        [Fact]
        public void GetFrequency_ReturnsZeroForUnknownShape()
        {
            var histogram = new ShapeHistogram();
            var shape = new[] { 10, 20, 30 };

            var freq = histogram.GetFrequency(shape);

            Assert.Equal(0.0, freq);
        }

        [Fact]
        public void GetTopShapes_ReturnsCorrectOrder()
        {
            var histogram = new ShapeHistogram();

            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 5, 10, 15 });
            histogram.AddSample(new[] { 5, 10, 15 });
            histogram.AddSample(new[] { 1, 2, 3 });

            var topShapes = histogram.GetTopShapes(3);

            Assert.Equal(3, topShapes.Count);
            Assert.Equal(3, topShapes[0].count);
            Assert.Equal(2, topShapes[1].count);
            Assert.Equal(1, topShapes[2].count);
        }

        [Fact]
        public void GetTopShapes_RespectsLimit()
        {
            var histogram = new ShapeHistogram();

            for (int i = 0; i < 10; i++)
            {
                histogram.AddSample(new[] { i, i * 2, i * 3 });
            }

            var topShapes = histogram.GetTopShapes(5);

            Assert.Equal(5, topShapes.Count);
        }

        [Fact]
        public void GetProbability_ReturnsSameAsFrequency()
        {
            var histogram = new ShapeHistogram();
            var shape = new[] { 10, 20, 30 };

            histogram.AddSample(shape);
            histogram.AddSample(shape);
            histogram.AddSample(new[] { 1, 2, 3 });

            var prob = histogram.GetProbability(shape);

            Assert.Equal(2.0 / 3.0, prob, 3);
        }

        [Fact]
        public void ToReport_GeneratesCorrectFormat()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 5, 10, 15 });

            var report = histogram.ToReport();

            Assert.Contains("Shape Histogram Report", report);
            Assert.Contains("Total Samples: 3", report);
            Assert.Contains("Unique Shapes: 2", report);
            Assert.Contains("[10, 20, 30]", report);
            Assert.Contains("[5, 10, 15]", report);
        }
    }
}
