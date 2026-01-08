using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.Exceptions;
using MLFramework.Core;

namespace MLFramework.Tests.Exceptions
{
    /// <summary>
    /// Unit tests for the ShapeMismatchException class.
    /// </summary>
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
            CollectionAssert.AreEqual(inputShapes, exception.InputShapes);
            CollectionAssert.AreEqual(expectedShapes, exception.ExpectedShapes);
            Assert.AreEqual(problemDescription, exception.ProblemDescription);
            CollectionAssert.AreEqual(suggestedFixes, exception.SuggestedFixes);
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
            Assert.AreEqual(OperationType.Conv2D, exception.OperationType);
            Assert.IsNull(exception.SuggestedFixes);
            Assert.IsFalse(exception.BatchSize.HasValue);
            Assert.IsNull(exception.PreviousLayerContext);
        }

        [Test]
        public void Constructor_WithBatchSize_SetsBatchSizeCorrectly()
        {
            // Arrange & Act
            var exception = new ShapeMismatchException(
                "layer1",
                OperationType.MatrixMultiply,
                new[] { new long[] { 32, 256 } },
                new[] { new long[] { 32, 10 } },
                "Shape error",
                suggestedFixes: null,
                batchSize: 32);

            // Assert
            Assert.IsNotNull(exception);
            Assert.AreEqual(32, exception.BatchSize);
            Assert.IsTrue(exception.BatchSize.HasValue);
        }

        [Test]
        public void Constructor_WithPreviousLayerContext_SetsContextCorrectly()
        {
            // Arrange & Act
            var previousContext = "encoder.fc1 [32, 256]";
            var exception = new ShapeMismatchException(
                "layer1",
                OperationType.Conv2D,
                new[] { new long[] { 32, 3, 224, 224 } },
                new[] { new long[] { 32, 64, 224, 224 } },
                "Shape error",
                previousLayerContext: previousContext);

            // Assert
            Assert.IsNotNull(exception);
            Assert.AreEqual(previousContext, exception.PreviousLayerContext);
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
            Assert.IsTrue(report.Contains("Batch size: 32"));
        }

        [Test]
        public void GetDiagnosticReport_ContainsInputShapes()
        {
            // Arrange
            var exception = new ShapeMismatchException(
                "test_layer",
                OperationType.MatrixMultiply,
                new[] { new long[] { 32, 256 }, new long[] { 128, 10 } },
                new[] { new long[] { 32, 10 } },
                "Inner dimension mismatch");

            // Act
            var report = exception.GetDiagnosticReport();

            // Assert
            Assert.IsTrue(report.Contains("Input shape"));
            Assert.IsTrue(report.Contains("[32, 256]"));
            Assert.IsTrue(report.Contains("Weight shape"));
            Assert.IsTrue(report.Contains("[128, 10]"));
        }

        [Test]
        public void GetDiagnosticReport_WithoutSuggestedFixes_DoesNotContainFixesSection()
        {
            // Arrange
            var exception = new ShapeMismatchException(
                "test_layer",
                OperationType.Conv2D,
                new[] { new long[] { 32, 3, 224, 224 } },
                new[] { new long[] { 32, 64, 224, 224 } },
                "Channel mismatch");

            // Act
            var report = exception.GetDiagnosticReport();

            // Assert
            Assert.IsFalse(report.Contains("Suggested fixes"));
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

        [Test]
        public void Exception_Message_ContainsOperationName()
        {
            // Arrange & Act
            var exception = new ShapeMismatchException(
                "conv1",
                OperationType.Conv2D,
                new[] { new long[] { 32, 3, 224, 224 } },
                new[] { new long[] { 32, 64, 224, 224 } },
                "Channel mismatch");

            // Assert
            Assert.IsTrue(exception.Message.Contains("Conv2D"));
        }

        [Test]
        public void Constructor_WithNullLayerName_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
            {
                new ShapeMismatchException(
                    null,
                    OperationType.MatrixMultiply,
                    new[] { new long[] { 32, 256 } },
                    new[] { new long[] { 32, 10 } },
                    "Error");
            });
        }

        [Test]
        public void Constructor_WithNullInputShapes_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
            {
                new ShapeMismatchException(
                    "layer1",
                    OperationType.MatrixMultiply,
                    null,
                    new[] { new long[] { 32, 10 } },
                    "Error");
            });
        }

        [Test]
        public void Constructor_WithNullProblemDescription_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
            {
                new ShapeMismatchException(
                    "layer1",
                    OperationType.MatrixMultiply,
                    new[] { new long[] { 32, 256 } },
                    new[] { new long[] { 32, 10 } },
                    null);
            });
        }

        [Test]
        public void InputShapesProperty_IsReadOnly()
        {
            // Arrange
            var exception = new ShapeMismatchException(
                "layer1",
                OperationType.MatrixMultiply,
                new[] { new long[] { 32, 256 } },
                new[] { new long[] { 32, 10 } },
                "Error");

            // Act & Assert
            // The InputShapes property should be IReadOnlyList, not modifiable
            Assert.IsInstanceOf<IReadOnlyList<long[]>>(exception.InputShapes);
        }

        [Test]
        public void GetDiagnosticReport_ForConv2DOperation_ContainsConv2DSpecificInfo()
        {
            // Arrange
            var exception = new ShapeMismatchException(
                "conv1",
                OperationType.Conv2D,
                new[] { new long[] { 32, 3, 224, 224 } },
                new[] { new long[] { 64, 3, 3, 3 } },
                "Channel mismatch");

            // Act
            var report = exception.GetDiagnosticReport();

            // Assert
            Assert.IsTrue(report.Contains("Conv2D operation failed"));
            Assert.IsTrue(report.Contains("Input shape"));
            Assert.IsTrue(report.Contains("Kernel shape"));
        }

        [Test]
        public void GetDiagnosticReport_ForConcatOperation_ContainsConcatSpecificInfo()
        {
            // Arrange
            var exception = new ShapeMismatchException(
                "concat1",
                OperationType.Concat,
                new[] { new long[] { 32, 10 }, new long[] { 32, 20 } },
                new[] { new long[] { 32, 30 } },
                "Dimension mismatch");

            // Act
            var report = exception.GetDiagnosticReport();

            // Assert
            Assert.IsTrue(report.Contains("Concatenation failed"));
            Assert.IsTrue(report.Contains("Input 0 shape"));
            Assert.IsTrue(report.Contains("Input 1 shape"));
        }

        [Test]
        public void VisualizeShapeFlow_ThrowsNotImplementedException()
        {
            // Arrange
            var exception = new ShapeMismatchException(
                "layer1",
                OperationType.MatrixMultiply,
                new[] { new long[] { 32, 256 } },
                new[] { new long[] { 32, 10 } },
                "Error");

            // Act & Assert
            Assert.Throws<NotImplementedException>(() =>
            {
                exception.VisualizeShapeFlow("output.txt");
            });
        }

        [Test]
        public void GetDiagnosticReport_MultipleSuggestedFixes_AllAreIncluded()
        {
            // Arrange
            var fixes = new List<string>
            {
                "Fix 1",
                "Fix 2",
                "Fix 3"
            };
            var exception = new ShapeMismatchException(
                "layer1",
                OperationType.MatrixMultiply,
                new[] { new long[] { 32, 256 } },
                new[] { new long[] { 32, 10 } },
                "Error",
                suggestedFixes: fixes);

            // Act
            var report = exception.GetDiagnosticReport();

            // Assert
            Assert.IsTrue(report.Contains("1. Fix 1"));
            Assert.IsTrue(report.Contains("2. Fix 2"));
            Assert.IsTrue(report.Contains("3. Fix 3"));
        }

        [Test]
        public void GetDiagnosticReport_WithoutPreviousLayerContext_DoesNotContainContext()
        {
            // Arrange
            var exception = new ShapeMismatchException(
                "layer1",
                OperationType.MatrixMultiply,
                new[] { new long[] { 32, 256 } },
                new[] { new long[] { 32, 10 } },
                "Error");

            // Act
            var report = exception.GetDiagnosticReport();

            // Assert
            Assert.IsFalse(report.Contains("Previous layer output"));
        }

        [Test]
        public void Constructor_WithMultipleInputShapes_HandlesCorrectly()
        {
            // Arrange & Act
            var inputShapes = new[]
            {
                new long[] { 32, 10 },
                new long[] { 32, 20 },
                new long[] { 32, 30 }
            };
            var exception = new ShapeMismatchException(
                "concat_layer",
                OperationType.Concat,
                inputShapes,
                new[] { new long[] { 32, 60 } },
                "Dimension mismatch");

            // Assert
            Assert.AreEqual(3, exception.InputShapes.Count);
            CollectionAssert.AreEqual(inputShapes, exception.InputShapes);
        }
    }
}
