using MLFramework.Data.Collate;
using Xunit;

namespace MLFramework.Tests.Data.Collate;

/// <summary>
/// Tests for StackCollateFunction.
/// </summary>
public class StackCollateFunctionTests
{
    [Fact]
    public void Collate_Stacks2DArraysInto3DArray()
    {
        // Arrange
        var collator = new StackCollateFunction();
        var batch = new[]
        {
            new[] { new[] { 1.0f, 2.0f }, new[] { 3.0f, 4.0f } },
            new[] { new[] { 5.0f, 6.0f }, new[] { 7.0f, 8.0f } }
        };

        // Act
        var result = collator.Collate(batch) as float[,,];

        // Assert
        Assert.NotNull(result);
        Assert.Equal(2, result.GetLength(0)); // batch size
        Assert.Equal(2, result.GetLength(1)); // height
        Assert.Equal(2, result.GetLength(2)); // width

        Assert.Equal(1.0f, result[0, 0, 0]);
        Assert.Equal(2.0f, result[0, 0, 1]);
        Assert.Equal(3.0f, result[0, 1, 0]);
        Assert.Equal(4.0f, result[0, 1, 1]);
        Assert.Equal(5.0f, result[1, 0, 0]);
        Assert.Equal(6.0f, result[1, 0, 1]);
        Assert.Equal(7.0f, result[1, 1, 0]);
        Assert.Equal(8.0f, result[1, 1, 1]);
    }

    [Fact]
    public void Collate_NullBatch_ThrowsArgumentException()
    {
        // Arrange
        var collator = new StackCollateFunction();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => collator.Collate(null));
    }

    [Fact]
    public void Collate_EmptyBatch_ThrowsArgumentException()
    {
        // Arrange
        var collator = new StackCollateFunction();
        var batch = Array.Empty<float[][]>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => collator.Collate(batch));
    }

    [Fact]
    public void Collate_SingleSample_Returns3DArrayWithBatchSizeOne()
    {
        // Arrange
        var collator = new StackCollateFunction();
        var batch = new[]
        {
            new[] { new[] { 1.0f, 2.0f }, new[] { 3.0f, 4.0f } }
        };

        // Act
        var result = collator.Collate(batch) as float[,,];

        // Assert
        Assert.NotNull(result);
        Assert.Equal(1, result.GetLength(0)); // batch size
        Assert.Equal(2, result.GetLength(1)); // height
        Assert.Equal(2, result.GetLength(2)); // width

        Assert.Equal(1.0f, result[0, 0, 0]);
        Assert.Equal(4.0f, result[0, 1, 1]);
    }

    [Fact]
    public void Collate_InconsistentHeight_ThrowsArgumentException()
    {
        // Arrange
        var collator = new StackCollateFunction();
        var batch = new[]
        {
            new[] { new[] { 1.0f, 2.0f }, new[] { 3.0f, 4.0f } },
            new[] { new[] { 5.0f, 6.0f } } // Missing second row
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => collator.Collate(batch));
    }

    [Fact]
    public void Collate_InconsistentWidth_ThrowsArgumentException()
    {
        // Arrange
        var collator = new StackCollateFunction();
        var batch = new[]
        {
            new[] { new[] { 1.0f, 2.0f }, new[] { 3.0f, 4.0f } },
            new[] { new[] { 5.0f, 6.0f }, new[] { 7.0f } } // Missing second element in second row
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => collator.Collate(batch));
    }

    [Fact]
    public void Collate_LargeBatch_HandlesCorrectly()
    {
        // Arrange
        var collator = new StackCollateFunction();
        var batchSize = 100;
        var height = 32;
        var width = 32;
        var batch = new float[batchSize][][];

        for (int b = 0; b < batchSize; b++)
        {
            batch[b] = new float[height][];
            for (int h = 0; h < height; h++)
            {
                batch[b][h] = new float[width];
                for (int w = 0; w < width; w++)
                {
                    batch[b][h][w] = b * height * width + h * width + w;
                }
            }
        }

        // Act
        var result = collator.Collate(batch) as float[,,];

        // Assert
        Assert.NotNull(result);
        Assert.Equal(batchSize, result.GetLength(0));
        Assert.Equal(height, result.GetLength(1));
        Assert.Equal(width, result.GetLength(2));

        // Verify a few values
        Assert.Equal(0, result[0, 0, 0]);
        Assert.Equal(width - 1, result[0, 0, width - 1]);
        Assert.Equal((batchSize - 1) * height * width, result[batchSize - 1, 0, 0]);
    }

    [Fact]
    public void Collate_DifferentBatchSizes_AllWork()
    {
        // Arrange
        var collator = new StackCollateFunction();

        // Test with batch size 2
        var batch2 = new[]
        {
            new[] { new[] { 1.0f }, new[] { 2.0f } },
            new[] { new[] { 3.0f }, new[] { 4.0f } }
        };
        var result2 = collator.Collate(batch2) as float[,,];
        Assert.Equal(2, result2.GetLength(0));

        // Test with batch size 10
        var batch10 = new float[10][][];
        for (int i = 0; i < 10; i++)
        {
            batch10[i] = new[] { new[] { (float)i } };
        }
        var result10 = collator.Collate(batch10) as float[,,];
        Assert.Equal(10, result10.GetLength(0));
    }

    [Fact]
    public void Collate_SingleElement2DArrays_WorksCorrectly()
    {
        // Arrange
        var collator = new StackCollateFunction();
        var batch = new[]
        {
            new[] { new[] { 1.0f } },
            new[] { new[] { 2.0f } },
            new[] { new[] { 3.0f } }
        };

        // Act
        var result = collator.Collate(batch) as float[,,];

        // Assert
        Assert.NotNull(result);
        Assert.Equal(3, result.GetLength(0));
        Assert.Equal(1, result.GetLength(1));
        Assert.Equal(1, result.GetLength(2));

        Assert.Equal(1.0f, result[0, 0, 0]);
        Assert.Equal(2.0f, result[1, 0, 0]);
        Assert.Equal(3.0f, result[2, 0, 0]);
    }
}
