namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for OptimizerStateDict
/// </summary>
public class OptimizerStateDictTests
{
    [Fact]
    public void Create_WithDefaultValues_CreatesValidState()
    {
        // Act
        var state = OptimizerStateDict.Create(OptimizerType.Adam);

        // Assert
        Assert.Equal(OptimizerType.Adam, state.OptimizerType);
        Assert.Equal(0, state.Step);
        Assert.Equal(0.001f, state.LearningRate);
    }

    [Fact]
    public void Create_WithLearningRate_SetsCorrectValue()
    {
        // Act
        var state = OptimizerStateDict.Create(OptimizerType.SGD, 0.01f);

        // Assert
        Assert.Equal(OptimizerType.SGD, state.OptimizerType);
        Assert.Equal(0.01f, state.LearningRate);
    }

    [Fact]
    public void GetParameterState_WithExistingParameter_ReturnsCorrectState()
    {
        // Arrange
        var state = OptimizerStateDict.Create(OptimizerType.Adam);
        var weightTensor = new MockTensor(new long[] { 128, 256 }, TensorDataType.Float32, 131072);
        var momentumTensor = new MockTensor(new long[] { 128, 256 }, TensorDataType.Float32, 131072);

        state["weight_momentum"] = momentumTensor;
        state["weight_data"] = weightTensor;

        // Act
        var paramState = state.GetParameterState("weight");

        // Assert
        Assert.Equal(2, paramState.Count);
        Assert.Contains("momentum", paramState.Keys);
        Assert.Contains("data", paramState.Keys);
        Assert.Equal(momentumTensor, paramState["momentum"]);
        Assert.Equal(weightTensor, paramState["data"]);
    }

    [Fact]
    public void SetParameterState_AddsParameterStateCorrectly()
    {
        // Arrange
        var state = OptimizerStateDict.Create(OptimizerType.Adam);
        var paramState = new StateDict
        {
            ["momentum"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40),
            ["data"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        state.SetParameterState("weight", paramState);

        // Assert
        Assert.True(state.ContainsKey("weight_momentum"));
        Assert.True(state.ContainsKey("weight_data"));
    }

    [Fact]
    public void SetParameterState_ReplacesExistingParameterState()
    {
        // Arrange
        var state = OptimizerStateDict.Create(OptimizerType.Adam);
        var oldParamState = new StateDict
        {
            ["old_field"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };
        var newParamState = new StateDict
        {
            ["new_field"] = new MockTensor(new long[] { 10 }, TensorDataType.Float32, 40)
        };

        // Act
        state.SetParameterState("weight", oldParamState);
        state.SetParameterState("weight", newParamState);

        // Assert
        Assert.False(state.ContainsKey("weight_old_field"));
        Assert.True(state.ContainsKey("weight_new_field"));
    }

    [Theory]
    [InlineData(OptimizerType.SGD)]
    [InlineData(OptimizerType.Adam)]
    [InlineData(OptimizerType.AdamW)]
    [InlineData(OptimizerType.RMSprop)]
    [InlineData(OptimizerType.Adagrad)]
    public void Create_WithAllOptimizerTypes_CreatesValidState(OptimizerType type)
    {
        // Act
        var state = OptimizerStateDict.Create(type);

        // Assert
        Assert.Equal(type, state.OptimizerType);
    }
}
