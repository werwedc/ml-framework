using System;
using System.Linq;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using MLFramework.Autograd.Exceptions;
using Xunit;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Unit tests for the GradientValidator utility class.
/// </summary>
public class GradientValidatorTests
{
    #region Shape Validation Tests

    [Fact]
    public void ValidateGradientShape_WithMatchingShapes_ShouldPass()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 2, 2 });
        var input = new Tensor(new float[] { 5.0f, 6.0f, 7.0f, 8.0f }, new[] { 2, 2 });

        // Act & Assert
        var exception = Record.Exception(() => GradientValidator.ValidateGradientShape(gradient, input));
        Assert.Null(exception);
    }

    [Fact]
    public void ValidateGradientShape_WithMismatchingShapes_ShouldThrowGradientShapeException()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 2, 2 });
        var input = new Tensor(new float[] { 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f }, new[] { 2, 3 });

        // Act & Assert
        var exception = Assert.Throws<GradientShapeException>(
            () => GradientValidator.ValidateGradientShape(gradient, input, "weight"));
        
        Assert.Contains("weight", exception.ParameterName);
        // Check expected shape is [2, 3] (input shape)
        Assert.Equal(2, exception.ExpectedShape[0]);
        Assert.Equal(3, exception.ExpectedShape[1]);
        // Check actual shape is [2, 2] (gradient shape)
        Assert.Equal(2, exception.ActualShape[0]);
        Assert.Equal(2, exception.ActualShape[1]);
    }

    [Fact]
    public void ValidateGradientShape_WithNullGradient_ShouldThrowArgumentNullException()
    {
        // Arrange
        Tensor gradient = null;
        var input = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GradientValidator.ValidateGradientShape(gradient, input));
    }

    [Fact]
    public void ValidateGradientShape_WithNullInput_ShouldThrowArgumentNullException()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        Tensor input = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GradientValidator.ValidateGradientShape(gradient, input));
    }

    [Fact]
    public void AreShapesCompatible_WithIdenticalShapes_ReturnsTrue()
    {
        // Arrange
        var shape1 = new Tensor(new float[4], new[] { 2, 2 });
        var shape2 = new Tensor(new float[4], new[] { 2, 2 });

        // Act
        var result = GradientValidator.AreShapesCompatible(shape1, shape2);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void AreShapesCompatible_WithBroadcastableShapes_ReturnsTrue()
    {
        // Arrange
        var shape1 = new Tensor(new float[4], new[] { 1, 2 });
        var shape2 = new Tensor(new float[8], new[] { 4, 2 });

        // Act
        var result = GradientValidator.AreShapesCompatible(shape1, shape2);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void AreShapesCompatible_WithIncompatibleShapes_ReturnsFalse()
    {
        // Arrange
        var shape1 = new Tensor(new float[4], new[] { 2, 2 });
        var shape2 = new Tensor(new float[6], new[] { 2, 3 });

        // Act
        var result = GradientValidator.AreShapesCompatible(shape1, shape2);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void AreShapesCompatible_WithNullShapes_ReturnsFalse()
    {
        // Act
        var result1 = GradientValidator.AreShapesCompatible(null, new Tensor(new float[1], new[] { 1 }));
        var result2 = GradientValidator.AreShapesCompatible(new Tensor(new float[1], new[] { 1 }), null);

        // Assert
        Assert.False(result1);
        Assert.False(result2);
    }

    #endregion

    #region Type Validation Tests

    [Fact]
    public void ValidateGradientType_WithMatchingDtypes_ShouldPass()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
        var input = new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 }, false, DataType.Float32);

        // Act & Assert
        var exception = Record.Exception(() => GradientValidator.ValidateGradientType(gradient, input));
        Assert.Null(exception);
    }

    [Fact]
    public void ValidateGradientType_WithMismatchingDtypes_ShouldThrowGradientTypeException()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
        var input = new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 }, false, DataType.Float64);

        // Act & Assert
        var exception = Assert.Throws<GradientTypeException>(
            () => GradientValidator.ValidateGradientType(gradient, input, "bias"));
        
        Assert.Contains("bias", exception.ParameterName);
        Assert.Equal(DataType.Float64, exception.ExpectedDtype);
        Assert.Equal(DataType.Float32, exception.ActualDtype);
    }

    [Fact]
    public void ValidateGradientType_WithNullGradient_ShouldThrowArgumentNullException()
    {
        // Arrange
        Tensor gradient = null;
        var input = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GradientValidator.ValidateGradientType(gradient, input));
    }

    [Fact]
    public void ValidateGradientType_WithNullInput_ShouldThrowArgumentNullException()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        Tensor input = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GradientValidator.ValidateGradientType(gradient, input));
    }

    #endregion

    #region Array Validation Tests

    [Fact]
    public void ValidateGradients_WithCorrectNumberOfGradients_ShouldPass()
    {
        // Arrange
        var gradients = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }),
            new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 })
        };

        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 5.0f, 6.0f }, new[] { 2 }),
            new Tensor(new float[] { 7.0f, 8.0f }, new[] { 2 })
        };

        // Act & Assert
        var exception = Record.Exception(() => GradientValidator.ValidateGradients(gradients, inputs));
        Assert.Null(exception);
    }

    [Fact]
    public void ValidateGradients_WithTooFewGradients_ShouldThrowAggregateException()
    {
        // Arrange
        var gradients = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 })
        };

        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 5.0f, 6.0f }, new[] { 2 }),
            new Tensor(new float[] { 7.0f, 8.0f }, new[] { 2 })
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => GradientValidator.ValidateGradients(gradients, inputs));
    }

    [Fact]
    public void ValidateGradients_WithTooManyGradients_ShouldThrowAggregateException()
    {
        // Arrange
        var gradients = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }),
            new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 }),
            new Tensor(new float[] { 5.0f, 6.0f }, new[] { 2 })
        };

        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 7.0f, 8.0f }, new[] { 2 }),
            new Tensor(new float[] { 9.0f, 10.0f }, new[] { 2 })
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => GradientValidator.ValidateGradients(gradients, inputs));
    }

    [Fact]
    public void ValidateGradients_WithNullGradients_ShouldPass()
    {
        // Arrange
        var gradients = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }),
            null,
            new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 })
        };

        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 5.0f, 6.0f }, new[] { 2 }),
            new Tensor(new float[] { 7.0f, 8.0f }, new[] { 2 }),
            new Tensor(new float[] { 9.0f, 10.0f }, new[] { 2 })
        };

        // Act & Assert
        var exception = Record.Exception(() => GradientValidator.ValidateGradients(gradients, inputs));
        Assert.Null(exception);
    }

    [Fact]
    public void ValidateGradients_WithNullGradientsArray_ShouldThrowArgumentNullException()
    {
        // Arrange
        Tensor[] gradients = null;
        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 })
        };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GradientValidator.ValidateGradients(gradients, inputs));
    }

    [Fact]
    public void ValidateGradients_WithNullInputsArray_ShouldThrowArgumentNullException()
    {
        // Arrange
        var gradients = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 })
        };
        Tensor[] inputs = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GradientValidator.ValidateGradients(gradients, inputs));
    }

    #endregion

    #region NaN/Inf Detection Tests

    [Fact]
    public void ValidateGradientHasNoNaN_WithoutNaN_ShouldPass()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });

        // Act & Assert
        var exception = Record.Exception(() => GradientValidator.ValidateGradientHasNoNaN(gradient));
        Assert.Null(exception);
    }

    [Fact]
    public void ValidateGradientHasNoNaN_WithNaN_ShouldThrowGradientNaNException()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, float.NaN, 3.0f, 4.0f }, new[] { 4 });

        // Act & Assert
        var exception = Assert.Throws<GradientNaNException>(
            () => GradientValidator.ValidateGradientHasNoNaN(gradient, "weight"));
        
        Assert.Contains("weight", exception.ParameterName);
        Assert.Equal(1, exception.NaNIndex); // NaN is at index 1
    }

    [Fact]
    public void ValidateGradientHasNoNaN_WithNullGradient_ShouldThrowArgumentNullException()
    {
        // Arrange
        Tensor gradient = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GradientValidator.ValidateGradientHasNoNaN(gradient));
    }

    [Fact]
    public void ValidateGradientHasNoInf_WithoutInf_ShouldPass()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });

        // Act & Assert
        var exception = Record.Exception(() => GradientValidator.ValidateGradientHasNoInf(gradient));
        Assert.Null(exception);
    }

    [Fact]
    public void ValidateGradientHasNoInf_WithPositiveInfinity_ShouldThrowGradientInfException()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, float.PositiveInfinity, 4.0f }, new[] { 4 });

        // Act & Assert
        var exception = Assert.Throws<GradientInfException>(
            () => GradientValidator.ValidateGradientHasNoInf(gradient, "bias"));
        
        Assert.Contains("bias", exception.ParameterName);
        Assert.Equal(2, exception.InfIndex);
        Assert.True(exception.IsPositiveInfinity);
    }

    [Fact]
    public void ValidateGradientHasNoInf_WithNegativeInfinity_ShouldThrowGradientInfException()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, float.NegativeInfinity, 4.0f }, new[] { 4 });

        // Act & Assert
        var exception = Assert.Throws<GradientInfException>(
            () => GradientValidator.ValidateGradientHasNoInf(gradient, "bias"));
        
        Assert.Contains("bias", exception.ParameterName);
        Assert.Equal(2, exception.InfIndex);
        Assert.False(exception.IsPositiveInfinity);
    }

    [Fact]
    public void ValidateGradientHasNoInf_WithNullGradient_ShouldThrowArgumentNullException()
    {
        // Arrange
        Tensor gradient = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GradientValidator.ValidateGradientHasNoInf(gradient));
    }

    [Fact]
    public void ContainsNaN_WithoutNaN_ReturnsFalse()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });

        // Act
        var result = GradientValidator.ContainsNaN(gradient);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ContainsNaN_WithNaN_ReturnsTrue()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, float.NaN, 3.0f, 4.0f }, new[] { 4 });

        // Act
        var result = GradientValidator.ContainsNaN(gradient);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void ContainsInf_WithoutInf_ReturnsFalse()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });

        // Act
        var result = GradientValidator.ContainsInf(gradient);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ContainsInf_WithPositiveInf_ReturnsTrue()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, float.PositiveInfinity, 3.0f, 4.0f }, new[] { 4 });

        // Act
        var result = GradientValidator.ContainsInf(gradient);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void ContainsInf_WithNegativeInf_ReturnsTrue()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, float.NegativeInfinity, 3.0f, 4.0f }, new[] { 4 });

        // Act
        var result = GradientValidator.ContainsInf(gradient);

        // Assert
        Assert.True(result);
    }

    #endregion

    #region Gradient Comparison Utilities Tests

    [Fact]
    public void AreGradientsEqual_WithEqualGradients_ReturnsTrue()
    {
        // Arrange
        var grad1 = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });
        var grad2 = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });

        // Act
        var result = GradientValidator.AreGradientsEqual(grad1, grad2);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void AreGradientsEqual_WithApproximatelyEqualGradients_ReturnsTrue()
    {
        // Arrange
        var grad1 = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });
        var grad2 = new Tensor(new float[] { 1.000001f, 2.000001f, 3.000001f, 4.000001f }, new[] { 4 });

        // Act
        var result = GradientValidator.AreGradientsEqual(grad1, grad2, tolerance: 1e-5);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void AreGradientsEqual_WithDifferentGradients_ReturnsFalse()
    {
        // Arrange
        var grad1 = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });
        var grad2 = new Tensor(new float[] { 1.1f, 2.1f, 3.1f, 4.1f }, new[] { 4 });

        // Act
        var result = GradientValidator.AreGradientsEqual(grad1, grad2);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void AreGradientsEqual_WithDifferentShapes_ReturnsFalse()
    {
        // Arrange
        var grad1 = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var grad2 = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });

        // Act
        var result = GradientValidator.AreGradientsEqual(grad1, grad2);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void AreGradientsEqual_WithNullGradients_ReturnsFalse()
    {
        // Act
        var result1 = GradientValidator.AreGradientsEqual(null, new Tensor(new float[1], new[] { 1 }));
        var result2 = GradientValidator.AreGradientsEqual(new Tensor(new float[1], new[] { 1 }), null);

        // Assert
        Assert.False(result1);
        Assert.False(result2);
    }

    [Fact]
    public void GetGradientDifference_WithValidGradients_ReturnsDifferenceTensor()
    {
        // Arrange
        var grad1 = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });
        var grad2 = new Tensor(new float[] { 0.5f, 1.5f, 2.5f, 3.5f }, new[] { 4 });

        // Act
        var result = GradientValidator.GetGradientDifference(grad1, grad2);

        // Assert
        Assert.Equal(4, result.Size);
        Assert.Equal(0.5f, result.Data[0]);
        Assert.Equal(0.5f, result.Data[1]);
        Assert.Equal(0.5f, result.Data[2]);
        Assert.Equal(0.5f, result.Data[3]);
    }

    [Fact]
    public void GetGradientDifference_WithDifferentShapes_ShouldThrowArgumentException()
    {
        // Arrange
        var grad1 = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });
        var grad2 = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 4 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => GradientValidator.GetGradientDifference(grad1, grad2));
    }

    [Fact]
    public void GetGradientDifference_WithNullGradient_ShouldThrowArgumentNullException()
    {
        // Arrange
        Tensor grad1 = null;
        var grad2 = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GradientValidator.GetGradientDifference(grad1, grad2));
    }

    #endregion

    #region Aggregate Validation Tests

    [Fact]
    public void ValidateGradientsAggregate_WithValidGradients_ReturnsValidResult()
    {
        // Arrange
        var gradients = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }),
            new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 })
        };

        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 5.0f, 6.0f }, new[] { 2 }),
            new Tensor(new float[] { 7.0f, 8.0f }, new[] { 2 })
        };

        // Act
        var result = GradientValidator.ValidateGradientsAggregate(gradients, inputs);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void ValidateGradientsAggregate_WithMultipleValidationFailures_ReturnsAllErrors()
    {
        // Arrange
        var gradients = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32),
            new Tensor(new float[] { float.NaN, 4.0f }, new[] { 3, 2 }, false, DataType.Float64)
        };

        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 5.0f, 6.0f }, new[] { 2 }, false, DataType.Float32),
            new Tensor(new float[] { 7.0f, 8.0f, 9.0f, 10.0f }, new[] { 2, 2 }, false, DataType.Float32)
        };

        // Act
        var result = GradientValidator.ValidateGradientsAggregate(gradients, inputs, checkNaN: true, checkInf: false);

        // Assert
        Assert.False(result.IsValid);
        Assert.Equal(3, result.Errors.Count); // Shape mismatch, dtype mismatch, NaN
    }

    [Fact]
    public void ValidateGradientsAggregate_WithMismatchedCounts_ReturnsError()
    {
        // Arrange
        var gradients = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 })
        };

        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 5.0f, 6.0f }, new[] { 2 }),
            new Tensor(new float[] { 7.0f, 8.0f }, new[] { 2 })
        };

        // Act
        var result = GradientValidator.ValidateGradientsAggregate(gradients, inputs);

        // Assert
        Assert.False(result.IsValid);
        Assert.Single(result.Errors);
        Assert.Contains("does not match", result.Errors[0]);
    }

    [Fact]
    public void ValidateGradientsAggregate_WithNullGradientsArray_ShouldThrowArgumentNullException()
    {
        // Arrange
        Tensor[] gradients = null;
        var inputs = new Tensor[] { new Tensor(new float[1], new[] { 1 }) };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GradientValidator.ValidateGradientsAggregate(gradients, inputs));
    }

    [Fact]
    public void ValidateGradientsAggregate_WithNullInputsArray_ShouldThrowArgumentNullException()
    {
        // Arrange
        var gradients = new Tensor[] { new Tensor(new float[1], new[] { 1 }) };
        Tensor[] inputs = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GradientValidator.ValidateGradientsAggregate(gradients, inputs));
    }

    [Fact]
    public void ValidateGradientsAggregate_WithNaNDisabled_DoesNotCheckForNaN()
    {
        // Arrange
        var gradients = new Tensor[]
        {
            new Tensor(new float[] { float.NaN }, new[] { 1 })
        };

        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 1.0f }, new[] { 1 })
        };

        // Act
        var result = GradientValidator.ValidateGradientsAggregate(gradients, inputs, checkNaN: false);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void ValidateGradientsAggregate_WithInfDisabled_DoesNotCheckForInf()
    {
        // Arrange
        var gradients = new Tensor[]
        {
            new Tensor(new float[] { float.PositiveInfinity }, new[] { 1 })
        };

        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 1.0f }, new[] { 1 })
        };

        // Act
        var result = GradientValidator.ValidateGradientsAggregate(gradients, inputs, checkInf: false);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    #endregion
}
