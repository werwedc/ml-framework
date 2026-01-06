using System.Collections.Generic;
using MLFramework.Data.Collate;
using Xunit;

namespace MLFramework.Tests.Data.Collate;

/// <summary>
/// Tests for DictionaryCollateFunction.
/// </summary>
public class DictionaryCollateFunctionTests
{
    [Fact]
    public void Collate_CollatesDictionaryBatch()
    {
        // Arrange
        var collator = new DictionaryCollateFunction();
        var batch = new[]
        {
            new Dictionary<string, object> { { "image", new[] { new[] { 1.0f, 2.0f }, new[] { 3.0f, 4.0f } } } },
            new Dictionary<string, object> { { "image", new[] { new[] { 5.0f, 6.0f }, new[] { 7.0f, 8.0f } } } }
        };

        // Act
        var result = collator.Collate(batch) as Dictionary<string, object>;

        // Assert
        Assert.NotNull(result);
        Assert.True(result.ContainsKey("image"));

        var stackedImages = result["image"] as float[,,];
        Assert.NotNull(stackedImages);
        Assert.Equal(2, stackedImages.GetLength(0)); // batch size
        Assert.Equal(2, stackedImages.GetLength(1)); // height
        Assert.Equal(2, stackedImages.GetLength(2)); // width

        Assert.Equal(1.0f, stackedImages[0, 0, 0]);
        Assert.Equal(8.0f, stackedImages[1, 1, 1]);
    }

    [Fact]
    public void Collate_NullBatch_ThrowsArgumentException()
    {
        // Arrange
        var collator = new DictionaryCollateFunction();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => collator.Collate(null));
    }

    [Fact]
    public void Collate_EmptyBatch_ThrowsArgumentException()
    {
        // Arrange
        var collator = new DictionaryCollateFunction();
        var batch = Array.Empty<Dictionary<string, object>>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => collator.Collate(batch));
    }

    [Fact]
    public void Collate_SingleSample_ReturnsDictionaryWithSingleSample()
    {
        // Arrange
        var collator = new DictionaryCollateFunction();
        var batch = new[]
        {
            new Dictionary<string, object> { { "image", new[] { new[] { 1.0f, 2.0f }, new[] { 3.0f, 4.0f } } } }
        };

        // Act
        var result = collator.Collate(batch) as Dictionary<string, object>;

        // Assert
        Assert.NotNull(result);
        Assert.True(result.ContainsKey("image"));

        var stackedImages = result["image"] as float[,,];
        Assert.NotNull(stackedImages);
        Assert.Equal(1, stackedImages.GetLength(0)); // batch size
    }

    [Fact]
    public void Collate_MissingKey_ThrowsArgumentException()
    {
        // Arrange
        var collator = new DictionaryCollateFunction();
        var batch = new[]
        {
            new Dictionary<string, object> { { "image", new[] { new[] { 1.0f } } }, { "label", 0 } },
            new Dictionary<string, object> { { "image", new[] { new[] { 2.0f } } } } // Missing "label" key
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => collator.Collate(batch));
    }

    [Fact]
    public void Collate_MultipleKeys_CollatesAllKeys()
    {
        // Arrange
        var collator = new DictionaryCollateFunction();
        var batch = new[]
        {
            new Dictionary<string, object>
            {
                { "image", new[] { new[] { 1.0f, 2.0f }, new[] { 3.0f, 4.0f } } },
                { "label", 0 }
            },
            new Dictionary<string, object>
            {
                { "image", new[] { new[] { 5.0f, 6.0f }, new[] { 7.0f, 8.0f } } },
                { "label", 1 }
            }
        };

        // Act
        var result = collator.Collate(batch) as Dictionary<string, object>;

        // Assert
        Assert.NotNull(result);
        Assert.Equal(2, result.Count);
        Assert.True(result.ContainsKey("image"));
        Assert.True(result.ContainsKey("label"));

        var stackedImages = result["image"] as float[,,];
        Assert.NotNull(stackedImages);
        Assert.Equal(2, stackedImages.GetLength(0));

        var labels = result["label"] as object[];
        Assert.NotNull(labels);
        Assert.Equal(2, labels.Length);
        Assert.Equal(0, labels[0]);
        Assert.Equal(1, labels[1]);
    }

    [Fact]
    public void Collate_NonImageValues_ReturnsArrayOfValues()
    {
        // Arrange
        var collator = new DictionaryCollateFunction();
        var batch = new[]
        {
            new Dictionary<string, object> { { "label", 0 } },
            new Dictionary<string, object> { { "label", 1 } },
            new Dictionary<string, object> { { "label", 2 } }
        };

        // Act
        var result = collator.Collate(batch) as Dictionary<string, object>;

        // Assert
        Assert.NotNull(result);
        Assert.True(result.ContainsKey("label"));

        var labels = result["label"] as object[];
        Assert.NotNull(labels);
        Assert.Equal(3, labels.Length);
        Assert.Equal(0, labels[0]);
        Assert.Equal(1, labels[1]);
        Assert.Equal(2, labels[2]);
    }

    [Fact]
    public void Collate_MixedValueTypes_CollatesCorrectly()
    {
        // Arrange
        var collator = new DictionaryCollateFunction();
        var batch = new[]
        {
            new Dictionary<string, object>
            {
                { "image", new[] { new[] { 1.0f, 2.0f } } },
                { "label", 0 },
                { "name", "sample1" }
            },
            new Dictionary<string, object>
            {
                { "image", new[] { new[] { 3.0f, 4.0f } } },
                { "label", 1 },
                { "name", "sample2" }
            }
        };

        // Act
        var result = collator.Collate(batch) as Dictionary<string, object>;

        // Assert
        Assert.NotNull(result);
        Assert.Equal(3, result.Count);

        var images = result["image"] as float[,,];
        Assert.NotNull(images);
        Assert.Equal(2, images.GetLength(0));

        var labels = result["label"] as object[];
        Assert.NotNull(labels);
        Assert.Equal(2, labels.Length);

        var names = result["name"] as object[];
        Assert.NotNull(names);
        Assert.Equal(2, names.Length);
        Assert.Equal("sample1", names[0]);
        Assert.Equal("sample2", names[1]);
    }

    [Fact]
    public void Collate_LargeBatch_HandlesCorrectly()
    {
        // Arrange
        var collator = new DictionaryCollateFunction();
        var batchSize = 100;
        var batch = new Dictionary<string, object>[batchSize];

        for (int i = 0; i < batchSize; i++)
        {
            batch[i] = new Dictionary<string, object>
            {
                { "image", new[] { new[] { (float)i } } },
                { "label", i }
            };
        }

        // Act
        var result = collator.Collate(batch) as Dictionary<string, object>;

        // Assert
        Assert.NotNull(result);

        var images = result["image"] as float[,,];
        Assert.NotNull(images);
        Assert.Equal(batchSize, images.GetLength(0));
        Assert.Equal(0, images[0, 0, 0]);
        Assert.Equal(batchSize - 1, images[batchSize - 1, 0, 0]);

        var labels = result["label"] as object[];
        Assert.NotNull(labels);
        Assert.Equal(batchSize, labels.Length);
        Assert.Equal(0, labels[0]);
        Assert.Equal(batchSize - 1, labels[batchSize - 1]);
    }
}
