using Xunit;
using MLFramework.Compilation;
using MLFramework.Shapes;

namespace MLFramework.Tests.Compilation;

/// <summary>
/// Unit tests for ShapeVariantGenerator
/// </summary>
public class ShapeVariantGeneratorTests
{
    [Fact]
    public void Constructor_WithDefaultSeed_CreatesGenerator()
    {
        // Arrange & Act
        var generator = new ShapeVariantGenerator();

        // Assert
        Assert.NotNull(generator);
    }

    [Fact]
    public void Constructor_WithSeed_CreatesDeterministicGenerator()
    {
        // Arrange
        var generator1 = new ShapeVariantGenerator(42);
        var generator2 = new ShapeVariantGenerator(42);
        var shape = new SymbolicShape(
            new SymbolicDimension("batch", 32),
            new SymbolicDimension("seq_len", 128)
        );

        // Act
        var variants1 = generator1.GenerateVariants(shape, 5);
        var variants2 = generator2.GenerateVariants(shape, 5);

        // Assert
        Assert.Equal(variants1.Count, variants2.Count);
        for (int i = 0; i < variants1.Count; i++)
        {
            Assert.Equal(variants1[i], variants2[i]);
        }
    }

    [Fact]
    public void GenerateVariants_WithFullyKnownShape_ReturnsConcreteShapes()
    {
        // Arrange
        var generator = new ShapeVariantGenerator();
        var shape = new SymbolicShape(
            new SymbolicDimension("batch", 32),
            new SymbolicDimension("seq_len", 128),
            new SymbolicDimension("hidden", 512)
        );

        // Act
        var variants = generator.GenerateVariants(shape, 3);

        // Assert
        Assert.Equal(3, variants.Count);
        Assert.All(variants, v => Assert.Equal(new[] { 32, 128, 512 }, v));
    }

    [Fact]
    public void GenerateVariants_WithPartiallyKnownShape_GeneratesVariants()
    {
        // Arrange
        var generator = new ShapeVariantGenerator(42);
        var shape = new SymbolicShape(
            new SymbolicDimension("batch", 32),
            new SymbolicDimension("seq_len", null, 32, 256) // Bounded unknown
        );

        // Act
        var variants = generator.GenerateVariants(shape, 5);

        // Assert
        Assert.Equal(5, variants.Count);
        Assert.All(variants, v =>
        {
            Assert.Equal(2, v.Length);
            Assert.Equal(32, v[0]); // Known dimension
            Assert.InRange(v[1], 32, 256); // Within bounds
        });
    }

    [Fact]
    public void GenerateVariants_WithUnboundedShape_UsesHeuristics()
    {
        // Arrange
        var generator = new ShapeVariantGenerator();
        var shape = new SymbolicShape(
            new SymbolicDimension("batch_size", null),
            new SymbolicDimension("sequence_length", null)
        );

        // Act
        var variants = generator.GenerateVariants(shape, 1);

        // Assert
        Assert.Single(variants);
        Assert.Equal(2, variants[0].Length);
        Assert.Equal(32, variants[0][0]); // Default for batch
        Assert.Equal(128, variants[0][1]); // Default for sequence
    }

    [Fact]
    public void GenerateGrid_WithMultipleShapes_CreatesCartesianProduct()
    {
        // Arrange
        var generator = new ShapeVariantGenerator(42);
        var shape1 = new SymbolicShape(
            new SymbolicDimension("batch", 32),
            new SymbolicDimension("hidden", 512)
        );
        var shape2 = new SymbolicShape(
            new SymbolicDimension("seq_len", 128)
        );
        var samplesPerDim = new List<int> { 2, 3 };

        // Act
        var grid = generator.GenerateGrid(new List<SymbolicShape> { shape1, shape2 }, samplesPerDim);

        // Assert
        Assert.Equal(6, grid.Count); // 2 * 3 = 6 combinations
        Assert.All(grid, combination =>
        {
            Assert.Equal(2, combination.Count);
            Assert.Equal(2, combination[0].Length); // shape1 rank
            Assert.Equal(1, combination[1].Length); // shape2 rank
        });
    }

    [Fact]
    public void GenerateGrid_WithMismatchedLengths_ThrowsArgumentException()
    {
        // Arrange
        var generator = new ShapeVariantGenerator();
        var shapes = new List<SymbolicShape>
        {
            new(new SymbolicDimension("batch", 32))
        };
        var samplesPerDim = new List<int> { 2, 3 }; // Wrong length

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            generator.GenerateGrid(shapes, samplesPerDim));
    }

    [Fact]
    public void GenerateVariantsWithConstraints_AppliesFixedValues()
    {
        // Arrange
        var generator = new ShapeVariantGenerator();
        var shape = new SymbolicShape(
            new SymbolicDimension("batch", 32),
            new SymbolicDimension("seq_len", null, 64, 256)
        );
        var fixedValues = new Dictionary<string, int>
        {
            { "seq_len", 128 }
        };

        // Act
        var variants = generator.GenerateVariantsWithConstraints(shape, fixedValues);

        // Assert
        Assert.Single(variants);
        Assert.Equal(new[] { 32, 128 }, variants[0]);
    }

    [Fact]
    public void GenerateVariantsWithConstraints_WithoutFixedValues_GeneratesDefault()
    {
        // Arrange
        var generator = new ShapeVariantGenerator();
        var shape = new SymbolicShape(
            new SymbolicDimension("batch", 64),
            new SymbolicDimension("hidden", 512)
        );

        // Act
        var variants = generator.GenerateVariantsWithConstraints(shape, null);

        // Assert
        Assert.Single(variants);
        Assert.Equal(new[] { 64, 512 }, variants[0]);
    }

    [Fact]
    public void GenerateTypicalVariants_GeneratesMultipleScales()
    {
        // Arrange
        var generator = new ShapeVariantGenerator(42);
        var shape = new SymbolicShape(
            new SymbolicDimension("batch", 32),
            new SymbolicDimension("hidden", 256)
        );

        // Act
        var variants = generator.GenerateTypicalVariants(shape);

        // Assert
        Assert.Equal(4, variants.Count); // scales: 1, 2, 4, 8 (but limited by loop)
        Assert.True(variants[1][0] > variants[0][0]); // Scaled up
    }

    [Fact]
    public void GenerateVariants_WithCommonDimensionNames_UsesCorrectDefaults()
    {
        // Arrange
        var generator = new ShapeVariantGenerator();
        var shape = new SymbolicShape(
            new SymbolicDimension("batch_size", null),
            new SymbolicDimension("sequence_length", null),
            new SymbolicDimension("hidden_size", null),
            new SymbolicDimension("num_channels", null),
            new SymbolicDimension("num_features", null),
            new SymbolicDimension("num_heads", null)
        );

        // Act
        var variant = generator.GenerateVariants(shape, 1)[0];

        // Assert
        Assert.Equal(32, variant[0]);   // batch_size default
        Assert.Equal(128, variant[1]);  // sequence_length default
        Assert.Equal(512, variant[2]);  // hidden_size default
        Assert.Equal(64, variant[3]);   // num_channels default
        Assert.Equal(256, variant[4]);  // num_features default
        Assert.Equal(8, variant[5]);    // num_heads default
    }

    [Fact]
    public void GenerateVariants_WithCountZero_ReturnsEmptyList()
    {
        // Arrange
        var generator = new ShapeVariantGenerator();
        var shape = new SymbolicShape(new SymbolicDimension("batch", 32));

        // Act
        var variants = generator.GenerateVariants(shape, 0);

        // Assert
        Assert.Empty(variants);
    }

    [Fact]
    public void GenerateVariants_WithMultipleUnknowns_GeneratesUniqueVariants()
    {
        // Arrange
        var generator = new ShapeVariantGenerator(42);
        var shape = new SymbolicShape(
            new SymbolicDimension("batch", null, 16, 32),
            new SymbolicDimension("seq", null, 64, 128)
        );

        // Act
        var variants = generator.GenerateVariants(shape, 10);

        // Assert
        Assert.Equal(10, variants.Count);
        var uniqueVariants = variants.Distinct().ToList();
        Assert.Equal(10, uniqueVariants.Count); // All should be unique
    }
}
