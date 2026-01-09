using RitterFramework.Core;
using RitterFramework.Core.Diagnostics;
using RitterFramework.Core.Tensor;
using MLFramework.Exceptions;

namespace MLFramework.Tests.TestUtilities;

/// <summary>
/// Helper methods for diagnostic tests.
/// </summary>
public static class DiagnosticsTestHelpers
{
    /// <summary>
    /// Creates a test tensor with the specified shape.
    /// </summary>
    public static Tensor CreateTestTensor(long[] shape)
    {
        var intShape = shape.Select(s => (int)s).ToArray();
        var size = 1;
        foreach (var dim in intShape)
        {
            size *= dim;
        }
        var data = new float[size];
        return new Tensor(data, intShape);
    }

    /// <summary>
    /// Creates a test shape mismatch exception.
    /// </summary>
    public static ShapeMismatchException CreateTestException()
    {
        return new ShapeMismatchException(
            "test_layer",
            OperationType.MatrixMultiply,
            new[] { new long[] { 32, 256 } },
            new[] { new long[] { 128, 10 } },
            "Dimension mismatch",
            new List<string> { "Fix 1", "Fix 2" });
    }

    /// <summary>
    /// Asserts that shape diagnostics match expected values.
    /// </summary>
    public static void AssertShapeDiagnostics(
        ShapeDiagnosticsInfo diagnostics,
        OperationType expectedType,
        string expectedLayer,
        bool expectedValid)
    {
        Assert.NotNull(diagnostics);
        Assert.Equal(expectedType, diagnostics.OperationType);
        Assert.Equal(expectedLayer, diagnostics.LayerName);
        Assert.Equal(expectedValid, diagnostics.IsValid);
    }

    /// <summary>
    /// Creates a test shape diagnostics info object.
    /// </summary>
    public static ShapeDiagnosticsInfo CreateTestDiagnosticsInfo()
    {
        return new ShapeDiagnosticsInfo
        {
            OperationType = OperationType.MatrixMultiply,
            LayerName = "test_layer",
            InputShapes = new[] { new long[] { 32, 256 }, new long[] { 128, 10 } },
            ExpectedShapes = new[] { new long[] { 32, 256 }, new long[] { 256, 10 } },
            ActualOutputShape = new long[] { 32, 10 },
            IsValid = false,
            Errors = new List<string> { "Dimension mismatch" },
            RequirementsDescription = "Matrix multiplication"
        };
    }

    /// <summary>
    /// Creates multiple test tensors with different shapes.
    /// </summary>
    public static Tensor[] CreateMultipleTestTensors(long[][] shapes)
    {
        return shapes.Select(CreateTestTensor).ToArray();
    }

    /// <summary>
    /// Asserts that validation result matches expected state.
    /// </summary>
    public static void AssertValidationResult(
        ValidationResult result,
        bool expectedValid,
        string? expectedError = null)
    {
        Assert.NotNull(result);
        Assert.Equal(expectedValid, result.IsValid);

        if (expectedError != null)
        {
            Assert.NotNull(result.Errors);
            Assert.Contains(expectedError, result.Errors);
        }
    }

    /// <summary>
    /// Creates a valid matrix multiplication test case.
    /// </summary>
    public static (Tensor[], OperationType) CreateValidMatrixMultiplyTestCase()
    {
        var tensors = new[]
        {
            CreateTestTensor(new[] { 32L, 10L }),
            CreateTestTensor(new[] { 10L, 5L })
        };
        return (tensors, OperationType.MatrixMultiply);
    }

    /// <summary>
    /// Creates an invalid matrix multiplication test case.
    /// </summary>
    public static (Tensor[], OperationType) CreateInvalidMatrixMultiplyTestCase()
    {
        var tensors = new[]
        {
            CreateTestTensor(new[] { 32L, 256L }),
            CreateTestTensor(new[] { 128L, 10L })
        };
        return (tensors, OperationType.MatrixMultiply);
    }

    /// <summary>
    /// Creates a valid Conv2D test case.
    /// </summary>
    public static (Tensor[], OperationType, Dictionary<string, object>) CreateValidConv2DTestCase()
    {
        var tensors = new[]
        {
            CreateTestTensor(new[] { 32L, 3L, 224L, 224L }),
            CreateTestTensor(new[] { 64L, 3L, 3L, 3L })
        };
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 1, 1 } },
            { "padding", new[] { 0, 0 } }
        };
        return (tensors, OperationType.Conv2D, parameters);
    }

    /// <summary>
    /// Creates a valid Concat test case.
    /// </summary>
    public static (Tensor[], OperationType, Dictionary<string, object>) CreateValidConcatTestCase()
    {
        var tensors = new[]
        {
            CreateTestTensor(new[] { 32L, 10L }),
            CreateTestTensor(new[] { 32L, 20L })
        };
        var parameters = new Dictionary<string, object>
        {
            { "axis", 1 }
        };
        return (tensors, OperationType.Concat, parameters);
    }
}
