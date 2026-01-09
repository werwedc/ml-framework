# Technical Spec: Diagnostics Tests

## Overview
Create comprehensive test suites for all shape mismatch error reporting components. This includes unit tests for individual components and integration tests that verify the entire diagnostics pipeline.

## Requirements

### Test Structure

#### 1. ShapeMismatchException Tests
File: `tests/Exceptions/ShapeMismatchExceptionTests.cs`

```csharp
[TestFixture]
public class ShapeMismatchExceptionTests
{
    [Test]
    public void Constructor_WithAllParameters_CreatesExceptionCorrectly()
    {
        // Arrange
        var layerName = "test_layer";
        var operationType = OperationType.MatrixMultiply;
        var inputShapes = new[] { new long[] { 32, 256 } };
        var expectedShapes = new[] { new long[] { 32, 128 } };
        var problemDescription = "Dimension mismatch";
        var suggestedFixes = new List<string> { "Fix 1", "Fix 2" };

        // Act
        var exception = new ShapeMismatchException(
            layerName,
            operationType,
            inputShapes,
            expectedShapes,
            problemDescription,
            suggestedFixes);

        // Assert
        Assert.AreEqual(layerName, exception.LayerName);
        Assert.AreEqual(operationType, exception.OperationType);
        Assert.AreEqual(inputShapes, exception.InputShapes);
        Assert.AreEqual(expectedShapes, exception.ExpectedShapes);
        Assert.AreEqual(problemDescription, exception.ProblemDescription);
        Assert.AreEqual(suggestedFixes, exception.SuggestedFixes);
    }

    [Test]
    public void Constructor_WithMinimalParameters_CreatesExceptionCorrectly()
    {
        // Arrange & Act
        var exception = new ShapeMismatchException(
            "test_layer",
            OperationType.Conv2D,
            new[] { new long[] { 32, 3, 224, 224 } },
            new[] { new long[] { 32, 64, 224, 224 } },
            "Channel mismatch");

        // Assert
        Assert.IsNotNull(exception);
        Assert.AreEqual("test_layer", exception.LayerName);
        Assert.IsNull(exception.SuggestedFixes);
        Assert.IsFalse(exception.BatchSize.HasValue);
    }

    [Test]
    public void GetDiagnosticReport_GeneratesCorrectFormat()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "encoder.fc2",
            OperationType.MatrixMultiply,
            new[] { new long[] { 32, 256 } },
            new[] { new long[] { 128, 10 } },
            "Dimension 1 mismatch",
            new List<string> { "Suggestion 1", "Suggestion 2" },
            32,
            "encoder.fc1 [32, 256]");

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert
        Assert.IsNotNull(report);
        Assert.IsTrue(report.Contains("encoder.fc2"));
        Assert.IsTrue(report.Contains("MatrixMultiply"));
        Assert.IsTrue(report.Contains("Dimension 1 mismatch"));
        Assert.IsTrue(report.Contains("Suggestion 1"));
        Assert.IsTrue(report.Contains("Suggestion 2"));
        Assert.IsTrue(report.Contains("encoder.fc1"));
    }

    [Test]
    public void Exception_Message_IsGeneratedCorrectly()
    {
        // Arrange & Act
        var exception = new ShapeMismatchException(
            "layer1",
            OperationType.Concat,
            new[] { new long[] { 32, 10 } },
            new[] { new long[] { 32, 20 } },
            "Channel mismatch");

        // Assert
        Assert.IsTrue(exception.Message.Contains("layer1"));
        Assert.IsTrue(exception.Message.Contains("Concat"));
        Assert.IsTrue(exception.Message.Contains("Shape mismatch"));
    }
}
```

#### 2. OperationMetadataRegistry Tests
File: `tests/Diagnostics/OperationMetadataRegistryTests.cs`

```csharp
[TestFixture]
public class OperationMetadataRegistryTests
{
    private IOperationMetadataRegistry _registry;

    [SetUp]
    public void Setup()
    {
        _registry = new DefaultOperationMetadataRegistry();
    }

    [Test]
    public void RegisterOperation_RegistersCorrectly()
    {
        // Arrange
        var requirements = new OperationShapeRequirements
        {
            InputCount = 2,
            ExpectedDimensions = new[] { 2, 2 },
            Description = "Test operation"
        };

        // Act
        _registry.RegisterOperation(OperationType.MatrixMultiply, requirements);

        // Assert
        Assert.IsTrue(_registry.IsRegistered(OperationType.MatrixMultiply));
        Assert.IsNotNull(_registry.GetRequirements(OperationType.MatrixMultiply));
    }

    [Test]
    public void ValidateShapes_MatrixMultiplyValidShapes_ReturnsSuccess()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 10, 5 } };

        // Act
        var result = _registry.ValidateShapes(
            OperationType.MatrixMultiply,
            inputShapes);

        // Assert
        Assert.IsTrue(result.IsValid);
        Assert.IsEmpty(result.Errors);
    }

    [Test]
    public void ValidateShapes_MatrixMultiplyInvalidShapes_ReturnsFailure()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 5, 10 } };

        // Act
        var result = _registry.ValidateShapes(
            OperationType.MatrixMultiply,
            inputShapes);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.IsNotEmpty(result.Errors);
        Assert.IsTrue(result.Errors[0].Contains("mismatch") ||
                        result.Errors[0].Contains("match"));
    }

    [Test]
    public void ValidateShapes_Conv2DChannelMismatch_ReturnsFailure()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 }, new long[] { 64, 64, 3, 3 } };
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 1, 1 } },
            { "padding", new[] { 0, 0 } }
        };

        // Act
        var result = _registry.ValidateShapes(
            OperationType.Conv2D,
            inputShapes,
            parameters);

        // Assert
        Assert.IsFalse(result.IsValid);
    }

    [Test]
    public void PreRegisteredOperations_AreAvailable()
    {
        // Assert
        Assert.IsTrue(_registry.IsRegistered(OperationType.MatrixMultiply));
        Assert.IsTrue(_registry.IsRegistered(OperationType.Conv2D));
        Assert.IsTrue(_registry.IsRegistered(OperationType.Concat));
    }
}
```

#### 3. ShapeInferenceEngine Tests
File: `tests/Diagnostics/ShapeInferenceEngineTests.cs`

```csharp
[TestFixture]
public class ShapeInferenceEngineTests
{
    private IShapeInferenceEngine _engine;

    [SetUp]
    public void Setup()
    {
        var registry = new DefaultOperationMetadataRegistry();
        _engine = new DefaultShapeInferenceEngine(registry);
    }

    [Test]
    public void InferOutputShape_MatrixMultiply2D_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 10, 5 } };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.MatrixMultiply,
            inputShapes);

        // Assert
        Assert.AreEqual(2, outputShape.Length);
        Assert.AreEqual(32, outputShape[0]);
        Assert.AreEqual(5, outputShape[1]);
    }

    [Test]
    public void InferOutputShape_MatrixMultiply3D_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 4, 32, 10 }, new long[] { 10, 5 } };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.MatrixMultiply,
            inputShapes);

        // Assert
        Assert.AreEqual(3, outputShape.Length);
        Assert.AreEqual(4, outputShape[0]);
        Assert.AreEqual(32, outputShape[1]);
        Assert.AreEqual(5, outputShape[2]);
    }

    [Test]
    public void InferOutputShape_Conv2D_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 }, new long[] { 64, 3, 3, 3 } };
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 1, 1 } },
            { "padding", new[] { 0, 0 } }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Conv2D,
            inputShapes,
            parameters);

        // Assert
        Assert.AreEqual(4, outputShape.Length);
        Assert.AreEqual(32, outputShape[0]); // batch
        Assert.AreEqual(64, outputShape[1]); // output channels
        Assert.AreEqual(222, outputShape[2]); // (224 - 3 + 0) / 1 + 1
        Assert.AreEqual(222, outputShape[3]);
    }

    [Test]
    public void InferOutputShape_Conv2DWithPadding_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 }, new long[] { 64, 3, 3, 3 } };
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 1, 1 } },
            { "padding", new[] { 1, 1 } }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Conv2D,
            inputShapes,
            parameters);

        // Assert
        Assert.AreEqual(224, outputShape[2]); // Same padding maintains size
        Assert.AreEqual(224, outputShape[3]);
    }

    [Test]
    public void InferOutputShape_MaxPool2D_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 64, 224, 224 } };
        var parameters = new Dictionary<string, object>
        {
            { "kernel_size", 2 },
            { "stride", 2 },
            { "padding", 0 }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.MaxPool2D,
            inputShapes,
            parameters);

        // Assert
        Assert.AreEqual(32, outputShape[0]);
        Assert.AreEqual(64, outputShape[1]);
        Assert.AreEqual(112, outputShape[2]); // 224 / 2
        Assert.AreEqual(112, outputShape[3]);
    }

    [Test]
    public void InferGraphShapes_SimpleLinearModel_ReturnsCorrectShapes()
    {
        // Arrange
        var graph = new ComputationGraph
        {
            Nodes = new Dictionary<string, OperationNode>
            {
                { "fc1", new OperationNode
                    {
                        Name = "fc1",
                        OperationType = OperationType.MatrixMultiply,
                        InputNames = new[] { "input" },
                        Parameters = new Dictionary<string, object>()
                    }},
                { "fc2", new OperationNode
                    {
                        Name = "fc2",
                        OperationType = OperationType.MatrixMultiply,
                        InputNames = new[] { "fc1" },
                        Parameters = new Dictionary<string, object>()
                    }}
            },
            Edges = new List<(string, string)>
            {
                ("input", "fc1"),
                ("fc1", "fc2")
            }
        };

        var inputShapes = new Dictionary<string, long[]>
        {
            { "input", new long[] { 32, 784 } },
            { "fc1", new long[] { 32, 256 } }, // Weight shape would be [784, 256]
            { "fc2", new long[] { 32, 10 } }    // Weight shape would be [256, 10]
        };

        // Act
        var result = _engine.InferGraphShapes(graph, inputShapes);

        // Assert
        Assert.AreEqual(3, result.Count);
        CollectionAssert.AreEqual(new long[] { 32, 784 }, result["input"]);
        CollectionAssert.AreEqual(new long[] { 32, 256 }, result["fc1"]);
        CollectionAssert.AreEqual(new long[] { 32, 10 }, result["fc2"]);
    }
}
```

#### 4. MLFrameworkDiagnostics Tests
File: `tests/Diagnostics/MLFrameworkDiagnosticsTests.cs`

```csharp
[TestFixture]
public class MLFrameworkDiagnosticsTests
{
    [TearDown]
    public void TearDown()
    {
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Test]
    public void EnableDiagnostics_SetsIsEnabledToTrue()
    {
        // Act
        MLFrameworkDiagnostics.EnableDiagnostics();

        // Assert
        Assert.IsTrue(MLFrameworkDiagnostics.IsEnabled);
    }

    [Test]
    public void DisableDiagnostics_SetsIsEnabledToFalse()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        // Act
        MLFrameworkDiagnostics.DisableDiagnostics();

        // Assert
        Assert.IsFalse(MLFrameworkDiagnostics.IsEnabled);
    }

    [Test]
    public void IsVerbose_SetCorrectly()
    {
        // Act
        MLFrameworkDiagnostics.EnableDiagnostics(verbose: true);

        // Assert
        Assert.IsTrue(MLFrameworkDiagnostics.IsVerbose);

        // Act
        MLFrameworkDiagnostics.EnableDiagnostics(verbose: false);

        // Assert
        Assert.IsFalse(MLFrameworkDiagnostics.IsVerbose);
    }

    [Test]
    public void GetShapeDiagnostics_ReturnsCorrectInfo()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = CreateTensor(new[] { 32L, 256L });
        var tensor2 = CreateTensor(new[] { 128L, 10L });

        // Act
        var diagnostics = MLFrameworkDiagnostics.GetShapeDiagnostics(
            OperationType.MatrixMultiply,
            new[] { tensor1, tensor2 },
            "test_layer");

        // Assert
        Assert.IsNotNull(diagnostics);
        Assert.AreEqual(OperationType.MatrixMultiply, diagnostics.OperationType);
        Assert.AreEqual("test_layer", diagnostics.LayerName);
        Assert.IsFalse(diagnostics.IsValid);
        Assert.IsNotEmpty(diagnostics.Errors);
    }

    [Test]
    public void GenerateSuggestedFixes_ReturnsAppropriateSuggestions()
    {
        // Arrange
        var diagnostics = new ShapeDiagnosticsInfo
        {
            OperationType = OperationType.MatrixMultiply,
            InputShapes = new[] { new long[] { 32, 256 }, new long[] { 128, 10 } },
            Errors = new List<string> { "Dimension mismatch" },
            IsValid = false
        };

        // Act
        var fixes = MLFrameworkDiagnostics.GenerateSuggestedFixes(diagnostics);

        // Assert
        Assert.IsNotEmpty(fixes);
        Assert.IsTrue(fixes.Any(f => f.Contains("weight") || f.Contains("matrix")));
    }

    private Tensor CreateTensor(long[] shape)
    {
        // Mock tensor creation for testing
        // In real implementation, this would create an actual Tensor object
        return new Tensor(shape);
    }
}
```

#### 5. ErrorReportingService Tests
File: `tests/Diagnostics/ErrorReportingServiceTests.cs`

```csharp
[TestFixture]
public class ErrorReportingServiceTests
{
    private IErrorReportingService _service;

    [SetUp]
    public void Setup()
    {
        _service = new ErrorReportingService();
    }

    [Test]
    public void CaptureContext_ThenGetCurrentContext_ReturnsSameContext()
    {
        // Arrange
        var context = new OperationExecutionContext
        {
            LayerName = "test_layer",
            OperationType = OperationType.MatrixMultiply,
            InputTensors = new[] { CreateTensor(new[] { 32L, 10L }) }
        };

        // Act
        _service.CaptureContext(context);
        var retrieved = _service.GetCurrentContext();

        // Assert
        Assert.IsNotNull(retrieved);
        Assert.AreEqual("test_layer", retrieved.LayerName);
        Assert.AreEqual(OperationType.MatrixMultiply, retrieved.OperationType);
    }

    [Test]
    public void ClearContext_MakesGetCurrentContextReturnNull()
    {
        // Arrange
        var context = new OperationExecutionContext
        {
            LayerName = "test_layer",
            OperationType = OperationType.Conv2D
        };
        _service.CaptureContext(context);

        // Act
        _service.ClearContext();
        var retrieved = _service.GetCurrentContext();

        // Assert
        Assert.IsNull(retrieved);
    }

    [Test]
    public void GenerateShapeMismatchException_WithValidContext_ReturnsException()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var context = new OperationExecutionContext
        {
            LayerName = "test_layer",
            OperationType = OperationType.MatrixMultiply,
            InputTensors = new[]
            {
                CreateTensor(new[] { 32L, 256L }),
                CreateTensor(new[] { 128L, 10L })
            }
        };
        _service.CaptureContext(context);

        // Act
        var exception = _service.GenerateShapeMismatchException("Dimension mismatch");

        // Assert
        Assert.IsNotNull(exception);
        Assert.IsInstanceOf<ShapeMismatchException>(exception);
        Assert.AreEqual("test_layer", exception.LayerName);
        Assert.AreEqual(OperationType.MatrixMultiply, exception.OperationType);

        MLFrameworkDiagnostics.DisableDiagnostics();
    }
}
```

#### 6. ShapeReportFormatter Tests
File: `tests/Diagnostics/ShapeReportFormatterTests.cs`

```csharp
[TestFixture]
public class ShapeReportFormatterTests
{
    [Test]
    public void Format_GeneratesCorrectReport()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "encoder.fc2",
            OperationType.MatrixMultiply,
            new[] { new long[] { 32, 256 } },
            new[] { new long[] { 128, 10 } },
            "Dimension 1 mismatch",
            new List<string> { "Fix 1", "Fix 2" },
            32,
            "encoder.fc1 [32, 256]");

        // Act
        var report = ShapeReportFormatter.Format(exception);

        // Assert
        Assert.IsNotNull(report);
        Assert.IsTrue(report.Contains("SHAPE MISMATCH"));
        Assert.IsTrue(report.Contains("encoder.fc2"));
        Assert.IsTrue(report.Contains("Input Shapes"));
        Assert.IsTrue(report.Contains("Dimension 1 mismatch"));
        Assert.IsTrue(report.Contains("Suggested Fixes"));
        Assert.IsTrue(report.Contains("Fix 1"));
        Assert.IsTrue(report.Contains("Fix 2"));
    }

    [Test]
    public void FormatSummary_GeneratesOneLineSummary()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "layer1",
            OperationType.Conv2D,
            new[] { new long[] { 32, 3, 224, 224 } },
            new[] { new long[] { 32, 64, 224, 224 } },
            "Channel mismatch");

        // Act
        var summary = ShapeReportFormatter.FormatSummary(exception);

        // Assert
        Assert.IsNotNull(summary);
        Assert.IsFalse(summary.Contains("\n"));
        Assert.IsTrue(summary.Contains("layer1"));
        Assert.IsTrue(summary.Contains("Conv2D"));
        Assert.IsTrue(summary.Contains("Channel mismatch"));
    }

    [Test]
    public void VisualizeShape_GeneratesCorrectASCII()
    {
        // Arrange
        var shape = new long[] { 32, 64, 224, 224 };

        // Act
        var visualization = ShapeReportFormatter.VisualizeShape(shape);

        // Assert
        Assert.IsNotNull(visualization);
        Assert.Contains("[32]", visualization);
        Assert.Contains("[64]", visualization);
        Assert.Contains("batch", visualization.ToLower());
        Assert.Contains("ch", visualization.ToLower());
    }
}
```

#### 7. OperationDiagnosticsHandler Tests
File: `tests/Diagnostics/OperationDiagnosticsHandlerTests.cs`

```csharp
[TestFixture]
public class MatrixMultiplyDiagnosticsHandlerTests
{
    private MatrixMultiplyDiagnosticsHandler _handler;

    [SetUp]
    public void Setup()
    {
        _handler = new MatrixMultiplyDiagnosticsHandler();
    }

    [Test]
    public void Validate_Valid2DShapes_ReturnsSuccess()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 10, 5 } };

        // Act
        var result = _handler.Validate(inputShapes, null);

        // Assert
        Assert.IsTrue(result.IsValid);
        Assert.IsEmpty(result.Errors);
    }

    [Test]
    public void Validate_InvalidInnerDimensions_ReturnsFailure()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 5, 10 } };

        // Act
        var result = _handler.Validate(inputShapes, null);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.IsNotEmpty(result.Errors);
        Assert.IsTrue(result.Errors[0].Contains("mismatch"));
    }

    [Test]
    public void GenerateErrorMessage_IncludesShapes()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 256 }, new long[] { 128, 10 } };

        // Act
        var message = _handler.GenerateErrorMessage(inputShapes, null, "test_layer");

        // Assert
        Assert.IsNotNull(message);
        Assert.Contains("[32, 256]", message);
        Assert.Contains("[128, 10]", message);
        Assert.Contains("test_layer", message);
    }

    [Test]
    public void GenerateSuggestions_IncludesRelevantFixes()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 256 }, new long[] { 128, 10 } };

        // Act
        var suggestions = _handler.GenerateSuggestions(inputShapes, null);

        // Assert
        Assert.IsNotEmpty(suggestions);
        Assert.IsTrue(suggestions.Any(s => s.Contains("transpose") ||
                                        s.Contains("configuration")));
    }
}

[TestFixture]
public class Conv2DDiagnosticsHandlerTests
{
    private Conv2DDiagnosticsHandler _handler;

    [SetUp]
    public void Setup()
    {
        _handler = new Conv2DDiagnosticsHandler();
    }

    [Test]
    public void ValidConv2DShapes_ReturnsSuccess()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 }, new long[] { 64, 3, 3, 3 } };
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 1, 1 } },
            { "padding", new[] { 0, 0 } }
        };

        // Act
        var result = _handler.Validate(inputShapes, parameters);

        // Assert
        Assert.IsTrue(result.IsValid);
    }

    [Test]
    public void ChannelMismatch_ReturnsFailure()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 }, new long[] { 64, 64, 3, 3 } };
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 1, 1 } },
            { "padding", new[] { 0, 0 } }
        };

        // Act
        var result = _handler.Validate(inputShapes, parameters);

        // Assert
        Assert.IsFalse(result.IsValid);
        Assert.IsTrue(result.Errors.Any(e => e.Contains("channel")));
    }
}
```

#### 8. Integration Tests
File: `tests/Integration/ShapeDiagnosticsIntegrationTests.cs`

```csharp
[TestFixture]
public class ShapeDiagnosticsIntegrationTests
{
    [Test]
    public void EndToEnd_InvalidMatrixMultiply_ThrowsShapeMismatchException()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();
        var tensor1 = CreateTensor(new[] { 32L, 256L });
        var tensor2 = CreateTensor(new[] { 128L, 10L });

        // Act & Assert
        Assert.Throws<ShapeMismatchException>(() =>
        {
            return tensor1.MatrixMultiply(tensor2, layerName: "test_layer");
        });

        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Test]
    public void EndToEnd_ExceptionContainsRichInformation()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();
        var tensor1 = CreateTensor(new[] { 32L, 256L });
        var tensor2 = CreateTensor(new[] { 128L, 10L });

        // Act
        ShapeMismatchException exception = null;
        try
        {
            tensor1.MatrixMultiply(tensor2, layerName: "encoder.fc2");
        }
        catch (ShapeMismatchException ex)
        {
            exception = ex;
        }

        // Assert
        Assert.IsNotNull(exception);
        Assert.AreEqual("encoder.fc2", exception.LayerName);
        Assert.AreEqual(OperationType.MatrixMultiply, exception.OperationType);
        Assert.IsNotNull(exception.SuggestedFixes);
        Assert.IsNotEmpty(exception.SuggestedFixes);
        Assert.IsTrue(exception.GetDiagnosticReport().Contains("Dimension"));

        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Test]
    public void DisabledDiagnostics_UsesBasicValidation()
    {
        // Arrange
        MLFrameworkDiagnostics.DisableDiagnostics();
        var tensor1 = CreateTensor(new[] { 32L, 256L });
        var tensor2 = CreateTensor(new[] { 128L, 10L });

        // Act & Assert
        // Should still throw, but with basic InvalidOperationException, not ShapeMismatchException
        Assert.Throws<InvalidOperationException>(() =>
        {
            tensor1.MatrixMultiply(tensor2, layerName: "test_layer");
        });
    }

    private Tensor CreateTensor(long[] shape)
    {
        // Mock tensor creation
        return new Tensor(shape);
    }
}
```

## Deliverables
- File: `tests/Exceptions/ShapeMismatchExceptionTests.cs`
- File: `tests/Diagnostics/OperationMetadataRegistryTests.cs`
- File: `tests/Diagnostics/ShapeInferenceEngineTests.cs`
- File: `tests/Diagnostics/MLFrameworkDiagnosticsTests.cs`
- File: `tests/Diagnostics/ErrorReportingServiceTests.cs`
- File: `tests/Diagnostics/ShapeReportFormatterTests.cs`
- File: `tests/Diagnostics/OperationDiagnosticsHandlerTests.cs`
- File: `tests/Integration/ShapeDiagnosticsIntegrationTests.cs`

## Additional Test Requirements

### Test Utilities
File: `tests/TestUtilities/DiagnosticsTestHelpers.cs`

```csharp
public static class DiagnosticsTestHelpers
{
    public static Tensor CreateTestTensor(long[] shape)
    {
        return new Tensor(shape);
    }

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

    public static void AssertShapeDiagnostics(
        ShapeDiagnosticsInfo diagnostics,
        OperationType expectedType,
        string expectedLayer,
        bool expectedValid)
    {
        Assert.AreEqual(expectedType, diagnostics.OperationType);
        Assert.AreEqual(expectedLayer, diagnostics.LayerName);
        Assert.AreEqual(expectedValid, diagnostics.IsValid);
    }
}
```

### Test Data
File: `tests/TestData/ShapeTestData.cs`

```csharp
public static class ShapeTestData
{
    // Valid MatrixMultiply shapes
    public static readonly (long[], long[])[] ValidMatrixMultiplyShapes = new[]
    {
        (new long[] { 32, 10 }, new long[] { 10, 5 }),
        (new long[] { 4, 32, 10 }, new long[] { 10, 5 }),
        (new long[] { 2, 4, 32, 10 }, new long[] { 10, 5 })
    };

    // Invalid MatrixMultiply shapes
    public static readonly (long[], long[])[] InvalidMatrixMultiplyShapes = new[]
    {
        (new long[] { 32, 10 }, new long[] { 5, 10 }),
        (new long[] { 32, 10 }, new long[] { 10, 5, 3 }), // Wrong dimensions
        (new long[] { 32 }, new long[] { 10 }) // 1D inputs
    };

    // Valid Conv2D shapes
    public static readonly (long[], long[], Dictionary<string, object>)[] ValidConv2DShapes = new[]
    {
        (new long[] { 32, 3, 224, 224 }, new long[] { 64, 3, 3, 3 }, new Dictionary<string, object> { { "stride", new[] { 1, 1 } }, { "padding", new[] { 0, 0 } } }),
        (new long[] { 16, 64, 112, 112 }, new long[] { 128, 64, 5, 5 }, new Dictionary<string, object> { { "stride", new[] { 1, 1 } }, { "padding", new[] { 2, 2 } } })
    };

    // Invalid Conv2D shapes
    public static readonly (long[], long[], Dictionary<string, object>)[] InvalidConv2DShapes = new[]
    {
        (new long[] { 32, 3, 224, 224 }, new long[] { 64, 64, 3, 3 }, new Dictionary<string, object> { { "stride", new[] { 1, 1 } }, { "padding", new[] { 0, 0 } } }), // Channel mismatch
        (new long[] { 32, 3, 224, 224 }, new long[] { 64, 3, 230, 230 }, new Dictionary<string, object> { { "stride", new[] { 1, 1 } }, { "padding", new[] { 0, 0 } } }) // Kernel larger than input
    };
}
```

## Notes
- Use NUnit or xUnit as the testing framework
- Mock Tensor objects for unit tests where actual tensor operations aren't needed
- Ensure tests are independent and can run in any order
- Use parameterized tests for testing multiple shape combinations
- Test both positive and negative cases
- Ensure tests are fast (avoid creating large tensors in tests)
- Use meaningful test names that describe what is being tested
- Consider using Moq or NSubstitute for mocking dependencies
