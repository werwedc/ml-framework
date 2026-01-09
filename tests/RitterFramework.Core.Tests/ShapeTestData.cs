namespace MLFramework.Tests.TestData;

/// <summary>
/// Test data for shape-related tests.
/// </summary>
public static class ShapeTestData
{
    #region Matrix Multiply Data

    /// <summary>
    /// Valid MatrixMultiply shapes.
    /// </summary>
    public static readonly (long[], long[])[] ValidMatrixMultiplyShapes = new[]
    {
        (new long[] { 32, 10 }, new long[] { 10, 5 }),
        (new long[] { 4, 32, 10 }, new long[] { 10, 5 }),
        (new long[] { 2, 4, 32, 10 }, new long[] { 10, 5 })
    };

    /// <summary>
    /// Invalid MatrixMultiply shapes.
    /// </summary>
    public static readonly (long[], long[])[] InvalidMatrixMultiplyShapes = new[]
    {
        (new long[] { 32, 10 }, new long[] { 5, 10 }),
        (new long[] { 32, 10 }, new long[] { 10, 5, 3 }), // Wrong dimensions
        (new long[] { 32 }, new long[] { 10 }) // 1D inputs
    };

    #endregion

    #region Conv2D Data

    /// <summary>
    /// Valid Conv2D shapes.
    /// </summary>
    public static readonly (long[], long[], Dictionary<string, object>)[] ValidConv2DShapes = new[]
    {
        (
            new long[] { 32, 3, 224, 224 },
            new long[] { 64, 3, 3, 3 },
            new Dictionary<string, object>
            {
                { "stride", new[] { 1, 1 } },
                { "padding", new[] { 0, 0 } }
            }
        ),
        (
            new long[] { 16, 64, 112, 112 },
            new long[] { 128, 64, 5, 5 },
            new Dictionary<string, object>
            {
                { "stride", new[] { 1, 1 } },
                { "padding", new[] { 2, 2 } }
            }
        )
    };

    /// <summary>
    /// Invalid Conv2D shapes.
    /// </summary>
    public static readonly (long[], long[], Dictionary<string, object>)[] InvalidConv2DShapes = new[]
    {
        (
            new long[] { 32, 3, 224, 224 },
            new long[] { 64, 64, 3, 3 },
            new Dictionary<string, object>
            {
                { "stride", new[] { 1, 1 } },
                { "padding", new[] { 0, 0 } }
            }
        ), // Channel mismatch
        (
            new long[] { 32, 3, 224, 224 },
            new long[] { 64, 3, 230, 230 },
            new Dictionary<string, object>
            {
                { "stride", new[] { 1, 1 } },
                { "padding", new[] { 0, 0 } }
            }
        ) // Kernel larger than input
    };

    #endregion

    #region Linear Data

    /// <summary>
    /// Valid Linear layer shapes.
    /// </summary>
    public static readonly (long[], long[])[] ValidLinearShapes = new[]
    {
        (new long[] { 32, 784 }, new long[] { 10, 784 }),
        (new long[] { 64, 512 }, new long[] { 256, 512 }),
        (new long[] { 128, 256 }, new long[] { 1000, 256 })
    };

    /// <summary>
    /// Invalid Linear layer shapes.
    /// </summary>
    public static readonly (long[], long[])[] InvalidLinearShapes = new[]
    {
        (new long[] { 32, 784 }, new long[] { 10, 512 }), // Weight dimension mismatch
        (new long[] { 64, 512 }, new long[] { 256, 256 }) // Weight dimension mismatch
    };

    #endregion

    #region Concat Data

    /// <summary>
    /// Valid Concat shapes.
    /// </summary>
    public static readonly (long[][], Dictionary<string, object>)[] ValidConcatShapes = new[]
    {
        (
            new[] { new long[] { 32, 10 }, new long[] { 32, 20 } },
            new Dictionary<string, object> { { "axis", 1 } }
        ),
        (
            new[] { new long[] { 32, 64 }, new long[] { 32, 64 }, new long[] { 32, 64 } },
            new Dictionary<string, object> { { "axis", 0 } }
        )
    };

    /// <summary>
    /// Invalid Concat shapes.
    /// </summary>
    public static readonly (long[][], Dictionary<string, object>)[] InvalidConcatShapes = new[]
    {
        (
            new[] { new long[] { 32, 128 }, new long[] { 64, 128 } },
            new Dictionary<string, object> { { "axis", 1 } }
        ) // Different batch sizes
    };

    #endregion

    #region Stack Data

    /// <summary>
    /// Valid Stack shapes.
    /// </summary>
    public static readonly (long[][], Dictionary<string, object>)[] ValidStackShapes = new[]
    {
        (
            new[] { new long[] { 32, 10 }, new long[] { 32, 10 }, new long[] { 32, 10 } },
            new Dictionary<string, object> { { "axis", 0 } }
        ),
        (
            new[] { new long[] { 64, 64 }, new long[] { 64, 64 } },
            new Dictionary<string, object> { { "axis", 1 } }
        )
    };

    /// <summary>
    /// Invalid Stack shapes.
    /// </summary>
    public static readonly (long[][], Dictionary<string, object>)[] InvalidStackShapes = new[]
    {
        (
            new[] { new long[] { 32, 10 }, new long[] { 64, 10 } },
            new Dictionary<string, object> { { "axis", 0 } }
        ) // Different shapes
    };

    #endregion

    #region Broadcast Data

    /// <summary>
    /// Valid Broadcast shapes.
    /// </summary>
    public static readonly (long[][])[] ValidBroadcastShapes = new[]
    {
        new[] { new long[] { 32, 1 }, new long[] { 32, 10 } },
        new[] { new long[] { 1, 10 }, new long[] { 32, 10 } },
        new[] { new long[] { 1 }, new long[] { 32, 10 } },
        new[] { new long[] { 32, 10 }, new long[] { 1, 10 } }
    };

    /// <summary>
    /// Invalid Broadcast shapes.
    /// </summary>
    public static readonly (long[][])[] InvalidBroadcastShapes = new[]
    {
        new[] { new long[] { 32, 10 }, new long[] { 20, 10 } }, // Incompatible batch sizes
        new[] { new long[] { 32, 10 }, new long[] { 32, 15 } } // Incompatible feature dimensions
    };

    #endregion

    #region Reshape Data

    /// <summary>
    /// Valid Reshape shapes.
    /// </summary>
    public static readonly (long[], long[])[] ValidReshapeShapes = new[]
    {
        (new long[] { 32, 10 }, new long[] { 320 }),
        (new long[] { 32, 10 }, new long[] { 16, 20 }),
        (new long[] { 32, 10 }, new long[] { 8, 5, 8 }),
        (new long[] { 32, 10 }, new long[] { -1, 5 }) // With inferred dimension
    };

    /// <summary>
    /// Invalid Reshape shapes.
    /// </summary>
    public static readonly (long[], long[])[] InvalidReshapeShapes = new[]
    {
        (new long[] { 32, 10 }, new long[] { 300 }), // Wrong element count
        (new long[] { 32, 10 }, new long[] { -1, -1 }) // Multiple inferred dimensions
    };

    #endregion

    #region Transpose Data

    /// <summary>
    /// Valid Transpose shapes.
    /// </summary>
    public static readonly (long[])[] ValidTransposeShapes = new[]
    {
        new long[] { 32, 10 },
        new long[] { 64, 64 },
        new long[] { 128, 256 },
        new long[] { 2, 3, 4 }
    };

    #endregion

    #region Common Test Tensors

    /// <summary>
    /// Common tensor shapes used in tests.
    /// </summary>
    public static readonly long[][] CommonTensorShapes = new[]
    {
        new long[] { 32, 10 },
        new long[] { 32, 64, 224, 224 },
        new long[] { 128, 256 },
        new long[] { 16, 32, 10 }
    };

    /// <summary>
    /// Small tensor shapes for quick tests.
    /// </summary>
    public static readonly long[][] SmallTensorShapes = new[]
    {
        new long[] { 2, 3 },
        new long[] { 1, 4 },
        new long[] { 3, 1 }
    };

    #endregion

    #region Edge Cases

    /// <summary>
    /// Edge case shapes (singletons, very large, etc.).
    /// </summary>
    public static readonly long[][] EdgeCaseShapes = new[]
    {
        new long[] { 1 }, // Scalar
        new long[] { 1, 1 }, // 1x1 matrix
        new long[] { 1, 10 }, // Row vector
        new long[] { 10, 1 }, // Column vector
        new long[] { 1024, 1024 } // Large matrix
    };

    /// <summary>
    /// Invalid shapes (negative or zero dimensions).
    /// </summary>
    public static readonly long[][] InvalidShapes = new[]
    {
        new long[] { -1, 10 }, // Negative dimension
        new long[] { 0, 10 }, // Zero dimension
        new long[] { } // Empty shape
    };

    #endregion
}
