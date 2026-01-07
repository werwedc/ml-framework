using MLFramework.Shapes;

namespace MLFramework.Compilation;

/// <summary>
/// Generates concrete shape variants from symbolic shapes for precompilation
/// </summary>
public class ShapeVariantGenerator
{
    private readonly Random _random;

    /// <summary>
    /// Creates a new shape variant generator
    /// </summary>
    public ShapeVariantGenerator(int? seed = null)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Generates concrete shape variants from a symbolic shape
    /// </summary>
    /// <param name="shape">The symbolic shape</param>
    /// <param name="count">Number of variants to generate</param>
    /// <returns>List of concrete shape arrays</returns>
    /// <exception cref="ArgumentException">Thrown when shape has unbounded dimensions without default values</exception>
    public List<int[]> GenerateVariants(SymbolicShape shape, int count)
    {
        var variants = new List<int[]>();

        for (int i = 0; i < count; i++)
        {
            var concreteShape = GenerateConcreteShape(shape);
            variants.Add(concreteShape);
        }

        return variants;
    }

    /// <summary>
    /// Generates a grid of shape combinations from multiple symbolic shapes
    /// </summary>
    /// <param name="shapes">List of symbolic shapes</param>
    /// <param name="samplesPerDim">Number of samples per dimension</param>
    /// <returns>List of concrete shape combinations</returns>
    public List<List<int[]>> GenerateGrid(List<SymbolicShape> shapes, List<int> samplesPerDim)
    {
        if (shapes.Count != samplesPerDim.Count)
        {
            throw new ArgumentException(
                $"Number of shapes ({shapes.Count}) must match number of samples per dimension ({samplesPerDim.Count})");
        }

        var grid = new List<List<int[]>>();
        var shapeVariants = new List<List<int[]>>();

        // Generate variants for each shape
        foreach (var (shape, samples) in shapes.Zip(samplesPerDim))
        {
            var variants = GenerateVariants(shape, samples);
            shapeVariants.Add(variants);
        }

        // Create Cartesian product of all shape combinations
        var combinations = CartesianProduct(shapeVariants);
        foreach (var combination in combinations)
        {
            grid.Add(combination.ToList());
        }

        return grid;
    }

    /// <summary>
    /// Generates variants with specific values for constrained dimensions
    /// </summary>
    /// <param name="shape">The symbolic shape</param>
    /// <param name="fixedValues">Dictionary mapping dimension names to fixed values</param>
    /// <returns>List of concrete shape arrays with fixed values applied</returns>
    public List<int[]> GenerateVariantsWithConstraints(
        SymbolicShape shape,
        Dictionary<string, int>? fixedValues = null)
    {
        var variants = new List<int[]>();
        var concreteShape = GenerateConcreteShape(shape);

        // Apply fixed values if provided
        if (fixedValues != null)
        {
            var fixedShape = (int[])concreteShape.Clone();

            for (int i = 0; i < shape.Rank; i++)
            {
                var dimName = shape.GetDimension(i).Name;
                if (fixedValues.TryGetValue(dimName, out var value))
                {
                    fixedShape[i] = value;
                }
            }

            variants.Add(fixedShape);
        }
        else
        {
            variants.Add(concreteShape);
        }

        return variants;
    }

    /// <summary>
    /// Generates a single concrete shape from a symbolic shape
    /// </summary>
    private int[] GenerateConcreteShape(SymbolicShape shape)
    {
        var concreteShape = new int[shape.Rank];

        for (int i = 0; i < shape.Rank; i++)
        {
            var dim = shape.GetDimension(i);

            if (dim.IsKnown())
            {
                // Use the known value
                concreteShape[i] = dim.Value!.Value;
            }
            else if (dim.IsBounded())
            {
                // Generate a random value within bounds
                var range = dim.MaxValue!.Value - dim.MinValue;
                concreteShape[i] = dim.MinValue + _random.Next(range + 1);
            }
            else
            {
                // Unbounded dimension - use a reasonable default or heuristic
                // This could be customized based on dimension name
                concreteShape[i] = GetDefaultValueForDimension(dim.Name);
            }
        }

        return concreteShape;
    }

    /// <summary>
    /// Gets a default value for an unbounded dimension based on heuristics
    /// </summary>
    private int GetDefaultValueForDimension(string dimensionName)
    {
        // Heuristics for common dimension names
        var lowerName = dimensionName.ToLower();

        return lowerName switch
        {
            string name when name.Contains("batch") => 32,
            string name when name.Contains("seq") || name.Contains("length") => 128,
            string name when name.Contains("hidden") || name.Contains("embed") => 512,
            string name when name.Contains("channel") => 64,
            string name when name.Contains("feature") => 256,
            string name when name.Contains("head") => 8,
            _ => 64 // Default for unknown dimensions
        };
    }

    /// <summary>
    /// Computes the Cartesian product of multiple lists
    /// </summary>
    private IEnumerable<IEnumerable<T>> CartesianProduct<T>(IEnumerable<IEnumerable<T>> sequences)
    {
        IEnumerable<IEnumerable<T>> emptyProduct = new[] { Enumerable.Empty<T>() };

        return sequences.Aggregate(
            emptyProduct,
            (accumulator, sequence) =>
                from accseq in accumulator
                from item in sequence
                select accseq.Concat(new[] { item })
        );
    }

    /// <summary>
    /// Generates a comprehensive set of shape variants for common use cases
    /// </summary>
    /// <param name="shape">The symbolic shape</param>
    /// <returns>List of concrete shapes covering typical scenarios</returns>
    public List<int[]> GenerateTypicalVariants(SymbolicShape shape)
    {
        var variants = new List<int[]>();

        // Generate small, medium, and large variants
        for (int scale = 1; scale <= 4; scale *= 2)
        {
            var scaledShape = GenerateConcreteShape(shape);

            for (int i = 0; i < scaledShape.Length; i++)
            {
                scaledShape[i] = Math.Max(1, scaledShape[i] * scale);
            }

            variants.Add(scaledShape);
        }

        return variants;
    }
}
