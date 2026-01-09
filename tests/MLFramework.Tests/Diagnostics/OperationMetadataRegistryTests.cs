using System;
using System.Collections.Generic;
using MLFramework.Core;
using MLFramework.Diagnostics;
using Xunit;

namespace MLFramework.Tests.Diagnostics
{
    /// <summary>
    /// Unit tests for OperationMetadataRegistry class.
    /// </summary>
    public class OperationMetadataRegistryTests
    {
        private DefaultOperationMetadataRegistry _registry;

        public OperationMetadataRegistryTests()
        {
            _registry = new DefaultOperationMetadataRegistry();
        }

        [Fact]
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
            Assert.True(_registry.IsRegistered(OperationType.MatrixMultiply));
            Assert.NotNull(_registry.GetRequirements(OperationType.MatrixMultiply));
        }

        [Fact]
        public void GetRequirements_ReturnsCorrectRequirements()
        {
            // Arrange & Act
            var requirements = _registry.GetRequirements(OperationType.MatrixMultiply);

            // Assert
            Assert.NotNull(requirements);
            Assert.Equal(2, requirements.InputCount);
            Assert.NotNull(requirements.ExpectedDimensions);
            Assert.Equal(2, requirements.ExpectedDimensions.Length);
        }

        [Fact]
        public void IsRegistered_UnregisteredOperation_ReturnsFalse()
        {
            // Act
            var isRegistered = _registry.IsRegistered(OperationType.Unknown);

            // Assert
            Assert.False(isRegistered);
        }

        [Fact]
        public void ValidateShapes_MatrixMultiplyValidShapes_ReturnsSuccess()
        {
            // Arrange
            var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 10, 5 } };

            // Act
            var result = _registry.ValidateShapes(
                OperationType.MatrixMultiply,
                inputShapes);

            // Assert
            Assert.True(result.IsValid);
            Assert.Empty(result.Errors);
        }

        [Fact]
        public void ValidateShapes_MatrixMultiplyInvalidShapes_ReturnsFailure()
        {
            // Arrange
            var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 5, 10 } };

            // Act
            var result = _registry.ValidateShapes(
                OperationType.MatrixMultiply,
                inputShapes);

            // Assert
            Assert.False(result.IsValid);
            Assert.NotEmpty(result.Errors);
        }

        [Fact]
        public void ValidateShapes_Conv2DValidShapes_ReturnsSuccess()
        {
            // Arrange
            var inputShapes = new[] { new long[] { 32, 3, 224, 224 }, new long[] { 64, 3, 3, 3 } };
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
            Assert.True(result.IsValid);
            Assert.Empty(result.Errors);
        }

        [Fact]
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
            Assert.False(result.IsValid);
            Assert.NotEmpty(result.Errors);
        }

        [Fact]
        public void ValidateShapes_WrongInputCount_ReturnsFailure()
        {
            // Arrange
            var inputShapes = new[] { new long[] { 32, 10 } }; // Only one input

            // Act
            var result = _registry.ValidateShapes(
                OperationType.MatrixMultiply,
                inputShapes);

            // Assert
            Assert.False(result.IsValid);
            Assert.NotEmpty(result.Errors);
            Assert.Contains("Expected", result.Errors[0]);
        }

        [Fact]
        public void ValidateShapes_WrongDimensions_ReturnsFailure()
        {
            // Arrange
            var inputShapes = new[] { new long[] { 32 }, new long[] { 10, 5 } }; // First input is 1D

            // Act
            var result = _registry.ValidateShapes(
                OperationType.MatrixMultiply,
                inputShapes);

            // Assert
            Assert.False(result.IsValid);
            Assert.NotEmpty(result.Errors);
        }

        [Fact]
        public void PreRegisteredOperations_AreAvailable()
        {
            // Assert
            Assert.True(_registry.IsRegistered(OperationType.MatrixMultiply));
            Assert.True(_registry.IsRegistered(OperationType.Conv2D));
            Assert.True(_registry.IsRegistered(OperationType.Concat));
            Assert.True(_registry.IsRegistered(OperationType.Stack));
            Assert.True(_registry.IsRegistered(OperationType.Flatten));
            Assert.True(_registry.IsRegistered(OperationType.Reshape));
            Assert.True(_registry.IsRegistered(OperationType.Transpose));
            Assert.True(_registry.IsRegistered(OperationType.Broadcast));
            Assert.True(_registry.IsRegistered(OperationType.MaxPool2D));
            Assert.True(_registry.IsRegistered(OperationType.AveragePool2D));
        }

        [Fact]
        public void ValidateShapes_ConcatValidShapes_ReturnsSuccess()
        {
            // Arrange
            var inputShapes = new[] 
            { 
                new long[] { 32, 10, 5 }, 
                new long[] { 32, 15, 5 },
                new long[] { 32, 8, 5 }
            };
            var parameters = new Dictionary<string, object>
            {
                { "axis", 1 }
            };

            // Act
            var result = _registry.ValidateShapes(
                OperationType.Concat,
                inputShapes,
                parameters);

            // Assert
            Assert.True(result.IsValid);
            Assert.Empty(result.Errors);
        }

        [Fact]
        public void ValidateShapes_ConcatInvalidShapes_ReturnsFailure()
        {
            // Arrange
            var inputShapes = new[] 
            { 
                new long[] { 32, 10, 5 }, 
                new long[] { 32, 15, 7 }  // Last dimension mismatch
            };
            var parameters = new Dictionary<string, object>
            {
                { "axis", 1 }
            };

            // Act
            var result = _registry.ValidateShapes(
                OperationType.Concat,
                inputShapes,
                parameters);

            // Assert
            Assert.False(result.IsValid);
            Assert.NotEmpty(result.Errors);
        }

        [Fact]
        public void ValidateShapes_StackValidShapes_ReturnsSuccess()
        {
            // Arrange
            var inputShapes = new[] 
            { 
                new long[] { 32, 10, 5 }, 
                new long[] { 32, 10, 5 },
                new long[] { 32, 10, 5 }
            };
            var parameters = new Dictionary<string, object>
            {
                { "axis", 0 }
            };

            // Act
            var result = _registry.ValidateShapes(
                OperationType.Stack,
                inputShapes,
                parameters);

            // Assert
            Assert.True(result.IsValid);
            Assert.Empty(result.Errors);
        }

        [Fact]
        public void ValidateShapes_StackInvalidShapes_ReturnsFailure()
        {
            // Arrange
            var inputShapes = new[] 
            { 
                new long[] { 32, 10, 5 }, 
                new long[] { 32, 15, 5 }  // Shape mismatch
            };
            var parameters = new Dictionary<string, object>
            {
                { "axis", 0 }
            };

            // Act
            var result = _registry.ValidateShapes(
                OperationType.Stack,
                inputShapes,
                parameters);

            // Assert
            Assert.False(result.IsValid);
            Assert.NotEmpty(result.Errors);
        }

        [Fact]
        public void ValidateShapes_ReshapeValidShapes_ReturnsSuccess()
        {
            // Arrange
            var inputShapes = new[] { new long[] { 32, 10, 5 } }; // 1600 elements
            var parameters = new Dictionary<string, object>
            {
                { "shape", new long[] { 1600, 1 } }
            };

            // Act
            var result = _registry.ValidateShapes(
                OperationType.Reshape,
                inputShapes,
                parameters);

            // Assert
            Assert.True(result.IsValid);
            Assert.Empty(result.Errors);
        }

        [Fact]
        public void ValidateShapes_ReshapeWithInferredDim_ReturnsSuccess()
        {
            // Arrange
            var inputShapes = new[] { new long[] { 32, 10, 5 } }; // 1600 elements
            var parameters = new Dictionary<string, object>
            {
                { "shape", new long[] { -1, 1 } }  // -1 inferred as 1600
            };

            // Act
            var result = _registry.ValidateShapes(
                OperationType.Reshape,
                inputShapes,
                parameters);

            // Assert
            Assert.True(result.IsValid);
            Assert.Empty(result.Errors);
        }

        [Fact]
        public void ValidateShapes_ReshapeInvalidShapes_ReturnsFailure()
        {
            // Arrange
            var inputShapes = new[] { new long[] { 32, 10, 5 } }; // 1600 elements
            var parameters = new Dictionary<string, object>
            {
                { "shape", new long[] { 500, 1 } }  // 500 elements, mismatch
            };

            // Act
            var result = _registry.ValidateShapes(
                OperationType.Reshape,
                inputShapes,
                parameters);

            // Assert
            Assert.False(result.IsValid);
            Assert.NotEmpty(result.Errors);
        }

        [Fact]
        public void ValidateShapes_ReshapeMultipleNegativeDims_ReturnsFailure()
        {
            // Arrange
            var inputShapes = new[] { new long[] { 32, 10, 5 } };
            var parameters = new Dictionary<string, object>
            {
                { "shape", new long[] { -1, -1, 1 } }  // Two -1 dimensions
            };

            // Act
            var result = _registry.ValidateShapes(
                OperationType.Reshape,
                inputShapes,
                parameters);

            // Assert
            Assert.False(result.IsValid);
            Assert.NotEmpty(result.Errors);
        }

        [Fact]
        public void ValidateShapes_BroadcastValidShapes_ReturnsSuccess()
        {
            // Arrange
            var inputShapes = new[] 
            { 
                new long[] { 3, 1, 24, 24 }, 
                new long[] { 1, 16, 1, 1 }
            };

            // Act
            var result = _registry.ValidateShapes(
                OperationType.Broadcast,
                inputShapes);

            // Assert
            Assert.True(result.IsValid);
            Assert.Empty(result.Errors);
        }

        [Fact]
        public void ValidateShapes_BroadcastInvalidShapes_ReturnsFailure()
        {
            // Arrange
            var inputShapes = new[] 
            { 
                new long[] { 3, 16, 24, 24 }, 
                new long[] { 5, 16, 24, 24 }  // First dimension incompatible
            };

            // Act
            var result = _registry.ValidateShapes(
                OperationType.Broadcast,
                inputShapes);

            // Assert
            Assert.False(result.IsValid);
            Assert.NotEmpty(result.Errors);
        }

        [Fact]
        public void RegisterOperation_OverwritesExistingOperation()
        {
            // Arrange
            var originalRequirements = _registry.GetRequirements(OperationType.MatrixMultiply);
            var newRequirements = new OperationShapeRequirements
            {
                InputCount = 3,
                Description = "New description"
            };

            // Act
            _registry.RegisterOperation(OperationType.MatrixMultiply, newRequirements);
            var updatedRequirements = _registry.GetRequirements(OperationType.MatrixMultiply);

            // Assert
            Assert.Equal(3, updatedRequirements.InputCount);
            Assert.Equal("New description", updatedRequirements.Description);
            Assert.NotEqual(originalRequirements.InputCount, updatedRequirements.InputCount);
        }

        [Fact]
        public void ValidationResult_Success_CreatesValidResult()
        {
            // Act
            var result = ValidationResult.Success();

            // Assert
            Assert.True(result.IsValid);
            Assert.Empty(result.Errors);
        }

        [Fact]
        public void ValidationResult_Failure_CreatesInvalidResult()
        {
            // Arrange
            var errors = new[] { "Error 1", "Error 2" };

            // Act
            var result = ValidationResult.Failure(errors);

            // Assert
            Assert.False(result.IsValid);
            Assert.Equal(2, result.Errors.Count);
            Assert.Contains("Error 1", result.Errors);
            Assert.Contains("Error 2", result.Errors);
        }

        [Fact]
        public void ValidationResult_AddError_MarksInvalid()
        {
            // Arrange
            var result = new ValidationResult();

            // Act
            result.AddError("Test error");

            // Assert
            Assert.False(result.IsValid);
            Assert.Single(result.Errors);
            Assert.Equal("Test error", result.Errors[0]);
        }

        [Fact]
        public void ValidationResult_AddWarning_DoesNotMarkInvalid()
        {
            // Arrange
            var result = new ValidationResult();

            // Act
            result.AddWarning("Test warning");

            // Assert
            Assert.True(result.IsValid);
            Assert.Single(result.Warnings);
            Assert.Equal("Test warning", result.Warnings[0]);
        }

        [Fact]
        public void DimensionConstraint_CreateMustMatch_CreatesCorrectConstraint()
        {
            // Act
            var constraint = DimensionConstraint.CreateMustMatch(1, 0);

            // Assert
            Assert.Equal(DimensionConstraint.ConstraintType.MustMatch, constraint.Type);
            Assert.Equal(1, constraint.TargetInputIndex);
            Assert.Equal(0, constraint.TargetDimensionIndex);
        }

        [Fact]
        public void DimensionConstraint_CreateMustEqual_CreatesCorrectConstraint()
        {
            // Act
            var constraint = DimensionConstraint.CreateMustEqual(64);

            // Assert
            Assert.Equal(DimensionConstraint.ConstraintType.MustEqual, constraint.Type);
            Assert.Equal(64, constraint.FixedValue);
        }

        [Fact]
        public void DimensionConstraint_CreateMustBeMultipleOf_CreatesCorrectConstraint()
        {
            // Act
            var constraint = DimensionConstraint.CreateMustBeMultipleOf(8);

            // Assert
            Assert.Equal(DimensionConstraint.ConstraintType.MustBeMultipleOf, constraint.Type);
            Assert.Equal(8, constraint.MultipleOf);
        }

        [Fact]
        public void DimensionConstraint_Validate_MustEqual_ValidatesCorrectly()
        {
            // Arrange
            var constraint = DimensionConstraint.CreateMustEqual(10);

            // Act & Assert
            Assert.True(constraint.Validate(10));
            Assert.False(constraint.Validate(5));
            Assert.False(constraint.Validate(15));
        }

        [Fact]
        public void DimensionConstraint_Validate_MustBeMultipleOf_ValidatesCorrectly()
        {
            // Arrange
            var constraint = DimensionConstraint.CreateMustBeMultipleOf(8);

            // Act & Assert
            Assert.True(constraint.Validate(8));
            Assert.True(constraint.Validate(16));
            Assert.True(constraint.Validate(24));
            Assert.False(constraint.Validate(10));
            Assert.False(constraint.Validate(15));
        }

        [Fact]
        public void DimensionConstraint_Validate_MustMatch_ValidatesCorrectly()
        {
            // Arrange
            var constraint = DimensionConstraint.CreateMustMatch(1, 0);
            long targetValue = 10;

            // Act & Assert
            Assert.True(constraint.Validate(10, targetValue));
            Assert.False(constraint.Validate(5, targetValue));
        }

        [Fact]
        public void OperationShapeRequirements_AddConstraint_AddsCorrectly()
        {
            // Arrange
            var requirements = new OperationShapeRequirements();

            // Act
            requirements.AddConstraint(0, 1, DimensionConstraint.CreateMustMatch(1, 0));

            // Assert
            Assert.True(requirements.DimensionConstraints.ContainsKey(0));
            Assert.True(requirements.DimensionConstraints[0].ContainsKey(1));
            Assert.Equal(DimensionConstraint.ConstraintType.MustMatch, requirements.DimensionConstraints[0][1].Type);
        }

        [Fact]
        public void OperationShapeRequirements_ValidateShapes_WithCustomValidator()
        {
            // Arrange
            var customValidatorCalled = false;
            var requirements = new OperationShapeRequirements
            {
                InputCount = 1,
                CustomValidator = (shapes, parameters) =>
                {
                    customValidatorCalled = true;
                    return ValidationResult.Success();
                }
            };

            // Act
            var result = requirements.ValidateShapes(new[] { new long[] { 32, 10 } });

            // Assert
            Assert.True(customValidatorCalled);
            Assert.True(result.IsValid);
        }
    }
}
