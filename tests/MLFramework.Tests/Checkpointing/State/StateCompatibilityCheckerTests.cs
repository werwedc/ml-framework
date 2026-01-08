namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for StateCompatibilityChecker
/// </summary>
public class StateCompatibilityCheckerTests
{
    private readonly StateCompatibilityChecker _checker;

    public StateCompatibilityCheckerTests()
    {
        _checker = new StateCompatibilityChecker();
    }

    [Fact]
    public void CheckCompatibility_WithIdenticalStates_ReturnsCompatible()
    {
        // Arrange
        var tensor = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800);
        var state1 = new StateDict { ["weight"] = tensor };
        var state2 = new StateDict { ["weight"] = tensor };

        // Act
        var result = _checker.CheckCompatibility(state1, state2);

        // Assert
        Assert.True(result.IsCompatible);
        Assert.Empty(result.Errors);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void CheckCompatibility_WithNullStates_ReturnsCompatible()
    {
        // Act
        var result = _checker.CheckCompatibility(null!, null!);

        // Assert
        Assert.True(result.IsCompatible);
    }

    [Fact]
    public void CheckCompatibility_WithOneNullState_ReturnsIncompatible()
    {
        // Arrange
        var state = new StateDict { ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40) };

        // Act
        var result1 = _checker.CheckCompatibility(state, null!);
        var result2 = _checker.CheckCompatibility(null!, state);

        // Assert
        Assert.False(result1.IsCompatible);
        Assert.False(result2.IsCompatible);
        Assert.Contains("null", result1.Errors[0]);
    }

    [Fact]
    public void CheckCompatibility_WithMissingKeys_AddsWarning()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var result = _checker.CheckCompatibility(state1, state2);

        // Assert
        Assert.True(result.IsCompatible); // Warnings don't make it incompatible
        Assert.True(result.HasWarnings);
        Assert.Contains("bias", result.Warnings[0]);
    }

    [Fact]
    public void CheckCompatibility_WithExtraKeys_AddsWarning()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["extra"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var result = _checker.CheckCompatibility(state1, state2);

        // Assert
        Assert.True(result.IsCompatible);
        Assert.True(result.HasWarnings);
        Assert.Contains("extra", result.Warnings[0]);
    }

    [Fact]
    public void CheckCompatibility_WithShapeMismatch_AddsError()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 30 }, TensorDataType.Float32, 1200)
        };

        // Act
        var result = _checker.CheckCompatibility(state1, state2);

        // Assert
        Assert.False(result.IsCompatible);
        Assert.Single(result.Errors);
        Assert.Contains("Shape mismatch", result.Errors[0]);
    }

    [Fact]
    public void CheckCompatibility_WithDataTypeMismatch_AddsError()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float64, 80)
        };

        // Act
        var result = _checker.CheckCompatibility(state1, state2);

        // Assert
        Assert.False(result.IsCompatible);
        Assert.Single(result.Errors);
        Assert.Contains("Data type mismatch", result.Errors[0]);
    }

    [Fact]
    public void CheckCompatibilityStrict_PromotesWarningsToErrors()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var state2 = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["extra"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var result = _checker.CheckCompatibilityStrict(state1, state2);

        // Assert
        Assert.False(result.IsCompatible);
        Assert.Empty(result.Warnings);
        Assert.NotEmpty(result.Errors);
    }

    [Fact]
    public void CheckOptimizerCompatibility_WithMatchingTypes_ReturnsCompatible()
    {
        // Arrange
        var state1 = OptimizerStateDict.Create(OptimizerType.Adam);
        var state2 = OptimizerStateDict.Create(OptimizerType.Adam);

        // Act
        var result = _checker.CheckOptimizerCompatibility(state1, state2);

        // Assert
        Assert.True(result.IsCompatible);
    }

    [Fact]
    public void CheckOptimizerCompatibility_WithDifferentTypes_AddsError()
    {
        // Arrange
        var state1 = OptimizerStateDict.Create(OptimizerType.Adam);
        var state2 = OptimizerStateDict.Create(OptimizerType.SGD);

        // Act
        var result = _checker.CheckOptimizerCompatibility(state1, state2);

        // Assert
        Assert.False(result.IsCompatible);
        Assert.Contains("Optimizer type mismatch", result.Errors[0]);
    }

    [Fact]
    public void CheckModelCompatibility_WithMatchingTypes_ReturnsCompatible()
    {
        // Arrange
        var state1 = ModelStateDict.Create("Transformer", 12);
        var state2 = ModelStateDict.Create("Transformer", 12);

        // Act
        var result = _checker.CheckModelCompatibility(state1, state2);

        // Assert
        Assert.True(result.IsCompatible);
    }

    [Fact]
    public void CheckModelCompatibility_WithDifferentTypes_AddsWarning()
    {
        // Arrange
        var state1 = ModelStateDict.Create("Transformer", 12);
        var state2 = ModelStateDict.Create("MLP", 2);

        // Act
        var result = _checker.CheckModelCompatibility(state1, state2);

        // Assert
        Assert.True(result.IsCompatible); // Warnings don't make it incompatible
        Assert.Contains("Model type mismatch", result.Warnings[0]);
    }

    [Fact]
    public void CheckModelCompatibility_WithDifferentLayerCount_AddsWarning()
    {
        // Arrange
        var state1 = ModelStateDict.Create("Transformer", 12);
        var state2 = ModelStateDict.Create("Transformer", 24);

        // Act
        var result = _checker.CheckModelCompatibility(state1, state2);

        // Assert
        Assert.True(result.IsCompatible);
        Assert.Contains("Layer count mismatch", result.Warnings[0]);
    }

    [Fact]
    public void CheckLoadCompatibility_WithMissingKeys_ReturnsIncompatible()
    {
        // Arrange
        var checkpointState = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var modelState = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var result = _checker.CheckLoadCompatibility(checkpointState, modelState, allowPartialLoad: false);

        // Assert
        Assert.False(result.IsCompatible);
        Assert.Contains("missing", result.Errors[0]);
    }

    [Fact]
    public void CheckLoadCompatibility_WithPartialLoadAllowed_ReturnsCompatibleWithWarning()
    {
        // Arrange
        var checkpointState = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var modelState = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["bias"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var result = _checker.CheckLoadCompatibility(checkpointState, modelState, allowPartialLoad: true);

        // Assert
        Assert.True(result.IsCompatible);
        Assert.True(result.HasWarnings);
    }

    [Fact]
    public void CheckLoadCompatibility_WithExtraKeysInCheckpoint_AddsWarning()
    {
        // Arrange
        var checkpointState = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["extra_param"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var modelState = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        var result = _checker.CheckLoadCompatibility(checkpointState, modelState);

        // Assert
        Assert.True(result.IsCompatible);
        Assert.True(result.HasWarnings);
        Assert.Contains("not present in model", result.Warnings[0]);
    }

    [Fact]
    public void CheckLoadCompatibility_WithShapeMismatch_ReturnsIncompatible()
    {
        // Arrange
        var checkpointState = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 20 }, TensorDataType.Float32, 800)
        };
        var modelState = new StateDict
        {
            ["weight"] = new MockTensor(new long[] { 10, 30 }, TensorDataType.Float32, 1200)
        };

        // Act
        var result = _checker.CheckLoadCompatibility(checkpointState, modelState);

        // Assert
        Assert.False(result.IsCompatible);
        Assert.Contains("Shape mismatch", result.Errors[0]);
    }
}
