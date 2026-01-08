using System;
using System.Linq;
using MLFramework.Profiling;
using Xunit;

namespace MLFramework.Tests.Profiling
{
    public class ShapeProfileOptimizerTests
    {
        [Fact]
        public void RecommendSpecializedShapes_ReturnsShapesAboveThreshold()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 5, 10, 15 });
            histogram.AddSample(new[] { 1, 2, 3 });

            var optimizer = new ShapeProfileOptimizer();
            var shapes = optimizer.RecommendSpecializedShapes("tensor1", histogram, threshold: 2);

            Assert.Single(shapes);
            Assert.Equal(new[] { 10, 20, 30 }, shapes[0]);
        }

        [Fact]
        public void RecommendSpecializedShapes_ReturnsEmptyWhenNoShapesMeetThreshold()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 5, 10, 15 });

            var optimizer = new ShapeProfileOptimizer();
            var shapes = optimizer.RecommendSpecializedShapes("tensor1", histogram, threshold: 10);

            Assert.Empty(shapes);
        }

        [Fact]
        public void RecommendSpecializedShapes_SortsByFrequency()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 5, 10, 15 });
            histogram.AddSample(new[] { 5, 10, 15 });
            histogram.AddSample(new[] { 1, 2, 3 });

            var optimizer = new ShapeProfileOptimizer();
            var shapes = optimizer.RecommendSpecializedShapes("tensor1", histogram, threshold: 2);

            Assert.Equal(2, shapes.Count);
            Assert.Equal(new[] { 10, 20, 30 }, shapes[0]); // 3 samples
            Assert.Equal(new[] { 5, 10, 15 }, shapes[1]); // 2 samples
        }

        [Fact]
        public void RecommendSpecializedShapes_ThrowsOnNullHistogram()
        {
            var optimizer = new ShapeProfileOptimizer();

            Assert.Throws<ArgumentNullException>(() =>
                optimizer.RecommendSpecializedShapes("tensor1", null!, 10));
        }

        [Fact]
        public void RecommendSpecializedShapes_ReturnsEmptyForEmptyHistogram()
        {
            var histogram = new ShapeHistogram();
            var optimizer = new ShapeProfileOptimizer();

            var shapes = optimizer.RecommendSpecializedShapes("tensor1", histogram, threshold: 2);

            Assert.Empty(shapes);
        }

        [Fact]
        public void ShouldRecompile_ReturnsTrueForRareShape()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 5, 10, 15 });

            var optimizer = new ShapeProfileOptimizer();
            var shouldRecompile = optimizer.ShouldRecompile(new[] { 5, 10, 15 }, histogram);

            Assert.True(shouldRecompile);
        }

        [Fact]
        public void ShouldRecompile_ReturnsFalseForCommonShape()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });

            var optimizer = new ShapeProfileOptimizer();
            var shouldRecompile = optimizer.ShouldRecompile(new[] { 10, 20, 30 }, histogram);

            Assert.False(shouldRecompile);
        }

        [Fact]
        public void ShouldRecompile_RespectsCustomThreshold()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });

            var optimizer = new ShapeProfileOptimizer();
            var shouldRecompile = optimizer.ShouldRecompile(new[] { 10, 20, 30 }, histogram, threshold: 0.8);

            Assert.True(shouldRecompile);
        }

        [Fact]
        public void ShouldRecompile_ThrowsOnNullHistogram()
        {
            var optimizer = new ShapeProfileOptimizer();

            Assert.Throws<ArgumentNullException>(() =>
                optimizer.ShouldRecompile(new[] { 10, 20, 30 }, null!));
        }

        [Fact]
        public void ShouldRecompile_ThrowsOnNullShape()
        {
            var histogram = new ShapeHistogram();
            var optimizer = new ShapeProfileOptimizer();

            Assert.Throws<ArgumentNullException>(() =>
                optimizer.ShouldRecompile(null!, histogram));
        }

        [Fact]
        public void GetOptimalPadding_ReturnsRoundedPowerOf2()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 10, 20 });
            histogram.AddSample(new[] { 15, 25 });
            histogram.AddSample(new[] { 20, 30 });

            var optimizer = new ShapeProfileOptimizer();
            var padding = optimizer.GetOptimalPadding(histogram);

            Assert.Equal(2, padding.Length);
            Assert.Equal(32, padding[0]); // 20 rounds to 32
            Assert.Equal(32, padding[1]); // 30 rounds to 32
        }

        [Fact]
        public void GetOptimalPadding_ThrowsOnNullHistogram()
        {
            var optimizer = new ShapeProfileOptimizer();

            Assert.Throws<ArgumentNullException>(() =>
                optimizer.GetOptimalPadding(null!));
        }

        [Fact]
        public void GetOptimalPadding_ReturnsEmptyForEmptyHistogram()
        {
            var histogram = new ShapeHistogram();
            var optimizer = new ShapeProfileOptimizer();

            var padding = optimizer.GetOptimalPadding(histogram);

            Assert.Empty(padding);
        }

        [Fact]
        public void GetOptimalBatchSize_ReturnsMostCommon()
        {
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 32, 10, 20 });
            histogram.AddSample(new[] { 64, 10, 20 });
            histogram.AddSample(new[] { 64, 10, 20 });

            var optimizer = new ShapeProfileOptimizer();
            var batchSize = optimizer.GetOptimalBatchSize(histogram, batchSizeIndex: 0);

            Assert.Equal(64, batchSize);
        }

        [Fact]
        public void GetOptimalBatchSize_ReturnsDefaultForEmptyHistogram()
        {
            var histogram = new ShapeHistogram();
            var optimizer = new ShapeProfileOptimizer();

            var batchSize = optimizer.GetOptimalBatchSize(histogram);

            Assert.Equal(32, batchSize);
        }

        [Fact]
        public void RecommendDynamicShapes_ReturnsTrueForHighDiversity()
        {
            var histogram = new ShapeHistogram();

            // Add many different shapes
            for (int i = 0; i < 10; i++)
            {
                histogram.AddSample(new[] { i, i * 2, i * 3 });
            }

            var optimizer = new ShapeProfileOptimizer();
            var recommend = optimizer.RecommendDynamicShapes(histogram);

            Assert.True(recommend);
        }

        [Fact]
        public void RecommendDynamicShapes_ReturnsFalseForLowDiversity()
        {
            var histogram = new ShapeHistogram();

            // Add the same shape many times
            for (int i = 0; i < 10; i++)
            {
                histogram.AddSample(new[] { 10, 20, 30 });
            }

            var optimizer = new ShapeProfileOptimizer();
            var recommend = optimizer.RecommendDynamicShapes(histogram);

            Assert.False(recommend);
        }

        [Fact]
        public void RecommendDynamicShapes_RespectsCustomThreshold()
        {
            var histogram = new ShapeHistogram();

            // Add some variety but not much
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 10, 20, 30 });
            histogram.AddSample(new[] { 15, 25, 35 });

            var optimizer = new ShapeProfileOptimizer();
            var recommend = optimizer.RecommendDynamicShapes(histogram, diversityThreshold: 0.1);

            // With a very low threshold, even moderate diversity should recommend dynamic
            Assert.True(recommend);
        }

        [Fact]
        public void RecommendDynamicShapes_ThrowsOnNullHistogram()
        {
            var optimizer = new ShapeProfileOptimizer();

            Assert.Throws<ArgumentNullException>(() =>
                optimizer.RecommendDynamicShapes(null!));
        }

        [Fact]
        public void RoundUpToPowerOf2_WorksWithVariousInputs()
        {
            var optimizer = new ShapeProfileOptimizer();

            // Test through private reflection or indirectly through GetOptimalPadding
            var histogram = new ShapeHistogram();
            histogram.AddSample(new[] { 1 });
            var padding = optimizer.GetOptimalPadding(histogram);
            Assert.Equal(1, padding[0]);

            histogram.AddSample(new[] { 5 });
            padding = optimizer.GetOptimalPadding(histogram);
            Assert.Equal(8, padding[0]);

            histogram.AddSample(new[] { 17 });
            padding = optimizer.GetOptimalPadding(histogram);
            Assert.Equal(32, padding[0]);
        }
    }
}
