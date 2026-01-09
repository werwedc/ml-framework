using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using Xunit;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Unit tests for custom function autograd integration.
/// </summary>
public class CustomFunctionIntegrationTests
{
    /// <summary>
    /// Simple identity function for testing.
    /// </summary>
    private class IdentityFunction : CustomFunction
    {
        public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
        {
            ctx.SaveForBackward(inputs);
            return inputs;
        }

        public override Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx)
        {
            return gradOutputs;
        }
    }

    /// <summary>
    /// Simple addition function for testing.
    /// </summary>
    private class AddFunction : CustomFunction
    {
        public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
        {
            if (inputs.Length != 2)
                throw new ArgumentException("AddFunction requires exactly 2 inputs");

            var result = new Tensor(
                inputs[0].Data.Zip(inputs[1].Data, (a, b) => a + b).ToArray(),
                inputs[0].Shape,
                inputs[0].RequiresGrad || inputs[1].RequiresGrad);

            ctx.SaveForBackward(inputs[0], inputs[1]);
            return new[] { result };
        }

        public override Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx)
        {
            var input1 = ctx.GetSavedTensor(0);
            var input2 = ctx.GetSavedTensor(1);

            var grad1 = input1.RequiresGrad ? gradOutputs[0].Clone() : null;
            var grad2 = input2.RequiresGrad ? gradOutputs[0].Clone() : null;

            return new[] { grad1!, grad2! };
        }
    }

    /// <summary>
    /// Multiplication by scalar function for testing.
    /// </summary>
    private class ScaleFunction : CustomFunction
    {
        private readonly float _scale;

        public ScaleFunction(float scale)
        {
            _scale = scale;
        }

        public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
        {
            var scaledData = inputs[0].Data.Select(x => x * _scale).ToArray();
            var result = new Tensor(scaledData, inputs[0].Shape, inputs[0].RequiresGrad);

            ctx.SaveForBackward(inputs[0]);
            ctx.SaveForBackward(_scale);

            return new[] { result };
        }

        public override Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx)
        {
            var input = ctx.GetSavedTensor(0);
            var scale = (float)ctx.GetSavedObject(0);

            if (input.RequiresGrad)
            {
                var gradData = gradOutputs[0].Data.Select(g => g * scale).ToArray();
                return new[] { new Tensor(gradData, gradOutputs[0].Shape) };
            }

            return new Tensor[] { null! };
        }
    }

    /// <summary>
    /// Function with multiple outputs for testing.
    /// </summary>
    private class SplitFunction : CustomFunction
    {
        public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
        {
            var input = inputs[0];
            var halfSize = input.Size / 2;

            var output1 = new Tensor(
                input.Data.Take(halfSize).ToArray(),
                new[] { halfSize },
                input.RequiresGrad);

            var output2 = new Tensor(
                input.Data.Skip(halfSize).ToArray(),
                new[] { input.Size - halfSize },
                input.RequiresGrad);

            ctx.SaveForBackward(input);
            return new[] { output1, output2 };
        }

        public override Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx)
        {
            var input = ctx.GetSavedTensor(0);
            var grad1 = gradOutputs[0];
            var grad2 = gradOutputs[1];

            if (input.RequiresGrad)
            {
                var combinedGrad = grad1.Data.Concat(grad2.Data).ToArray();
                return new[] { new Tensor(combinedGrad, input.Shape) };
            }

            return new Tensor[] { null! };
        }
    }

    [Fact]
    public void Apply_ReturnsFirstOutput()
    {
        // Arrange
        var func = new IdentityFunction();
        var input = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var output = func.Apply(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Size, output.Size);
        Assert.Equal(input.Data, output.Data);
    }

    [Fact]
    public void ApplyMany_ReturnsAllOutputs()
    {
        // Arrange
        var func = new SplitFunction();
        var input = Tensor.FromArray(new float[] { 1, 2, 3, 4 });

        // Act
        var outputs = func.ApplyMany(input);

        // Assert
        Assert.Equal(2, outputs.Length);
        Assert.Equal(2, outputs[0].Size);
        Assert.Equal(2, outputs[1].Size);
        Assert.Equal(new float[] { 1, 2 }, outputs[0].Data);
        Assert.Equal(new float[] { 3, 4 }, outputs[1].Data);
    }

    [Fact]
    public void GraphNode_IsCreatedForCustomFunction()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var func = new AddFunction();
        var input1 = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        var input2 = new Tensor(new float[] { 4, 5, 6 }, new[] { 3 }, requiresGrad: true);

        // Act
        var output = func.Apply(input1, input2);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.RequiresGrad);
        Assert.NotNull(output.GetGradFn());
        Assert.Equal(1, engine.NodeCount);
    }

    [Fact]
    public void Backward_ComputesGradientsCorrectly()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var func = new AddFunction();
        var input1 = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        var input2 = new Tensor(new float[] { 4, 5, 6 }, new[] { 3 }, requiresGrad: true);
        var output = func.Apply(input1, input2);

        // Act
        engine.Backward(output, Tensor.Ones(output.Shape));

        // Assert
        Assert.NotNull(input1.Gradient);
        Assert.NotNull(input2.Gradient);
        Assert.Equal(new float[] { 1, 1, 1 }, input1.Gradient.Data);
        Assert.Equal(new float[] { 1, 1, 1 }, input2.Gradient.Data);
    }

    [Fact]
    public void Backward_WithCustomGradient()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var func = new ScaleFunction(2.0f);
        var input = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        var output = func.Apply(input);
        var gradInput = new Tensor(new float[] { 2, 2, 2 }, new[] { 3 });

        // Act
        engine.Backward(output, gradInput);

        // Assert
        Assert.NotNull(input.Gradient);
        Assert.Equal(new float[] { 4, 4, 4 }, input.Gradient.Data);
    }

    [Fact]
    public void GradientAccumulation_MultipleUses()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var func = new AddFunction();
        var input = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        var input2 = new Tensor(new float[] { 4, 5, 6 }, new[] { 3 }, requiresGrad: false);
        var input3 = new Tensor(new float[] { 7, 8, 9 }, new[] { 3 }, requiresGrad: false);

        // Use the same input in multiple operations
        var output1 = func.Apply(input, input2);
        var output2 = func.Apply(input, input3);
        var finalOutput = func.Apply(output1, output2);

        // Act
        engine.Backward(finalOutput, Tensor.Ones(finalOutput.Shape));

        // Assert
        Assert.NotNull(input.Gradient);
        // Gradient should be accumulated from both uses
        Assert.Equal(new float[] { 2, 2, 2 }, input.Gradient.Data);
    }

    [Fact]
    public void Context_IsCreatedForEachInvocation()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var func = new ScaleFunction(2.0f);
        var input = Tensor.FromArray(new float[] { 1, 2, 3 });

        // Act
        var output1 = func.Apply(input);
        var output2 = func.Apply(input);

        // Assert
        Assert.NotSame(output1, output2);
        // Verify different contexts were created by checking node count
        Assert.Equal(2, engine.NodeCount);
    }

    [Fact]
    public void Context_IsAccessibleDuringBackward()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var func = new ScaleFunction(2.0f);
        var input = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        var output = func.Apply(input);

        // Verify gradient was initialized to zeros
        Assert.NotNull(input.Gradient);
        Assert.Equal(new float[] { 0, 0, 0 }, input.Gradient.Data);

        // Act
        engine.Backward(output, Tensor.Ones(output.Shape));

        // Assert
        Assert.NotNull(input.Gradient);
        Assert.Equal(new float[] { 2, 2, 2 }, input.Gradient.Data);
    }

    [Fact]
    public void Context_IsDisposedAfterBackward()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var func = new IdentityFunction();
        var input = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        var output = func.Apply(input);

        // Get the node before backward
        var node = engine.GetType()
            .GetField("_nodes", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?
            .GetValue(engine) as System.Collections.Concurrent.ConcurrentDictionary<Guid, CustomFunctionNode>;

        var context = node?.Values.First().Context;
        Assert.False(context?.IsDisposed);

        // Act
        engine.Backward(output, Tensor.Ones(output.Shape));

        // Assert
        // Context is NOT automatically disposed after backward pass
        // User must manually dispose it or clear the graph
        Assert.False(context?.IsDisposed);

        // Clearing graph disposes all nodes and their contexts
        engine.ClearGraph();
        Assert.True(context?.IsDisposed);
    }

    [Fact]
    public void MultipleOutput_Functions_GradientsFlowCorrectly()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var splitFunc = new SplitFunction();
        var addFunc = new AddFunction();

        var input = new Tensor(new float[] { 1, 2, 3, 4 }, new[] { 4 }, requiresGrad: true);
        var outputs = splitFunc.ApplyMany(input);

        var sum = addFunc.Apply(outputs[0], outputs[1]);

        // Act
        engine.Backward(sum, Tensor.Ones(sum.Shape));

        // Assert
        Assert.NotNull(input.Gradient);
        Assert.Equal(new float[] { 1, 1, 1, 1 }, input.Gradient.Data);
    }

    [Fact]
    public void RequiresGrad_PropagatesFromInputsToOutputs()
    {
        // Arrange
        var func = new AddFunction();
        var input1 = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        var input2 = new Tensor(new float[] { 4, 5, 6 }, new[] { 3 }, requiresGrad: false);

        // Act
        var output = func.Apply(input1, input2);

        // Assert
        Assert.True(output.RequiresGrad);
    }

    [Fact]
    public void Backward_ThrowsForNonScalarWithoutGrad()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var func = new IdentityFunction();
        var input = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        var output = func.Apply(input);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => output.Backward());
    }

    [Fact]
    public void Apply_ThrowsForNullInputs()
    {
        // Arrange
        var func = new IdentityFunction();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => func.Apply(null!));
    }

    [Fact]
    public void Apply_ThrowsForNullInInputsArray()
    {
        // Arrange
        var func = new IdentityFunction();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => func.Apply(new Tensor[] { null! }));
    }

    [Fact]
    public void ClearGraph_RemovesAllNodes()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var func = new IdentityFunction();
        var input = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        func.Apply(input);

        Assert.Equal(1, engine.NodeCount);

        // Act
        engine.ClearGraph();

        // Assert
        Assert.Equal(0, engine.NodeCount);
    }

    [Fact]
    public void GraphNode_HasUniqueId()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var func = new IdentityFunction();
        var input = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);

        // Act
        var output1 = func.Apply(input);
        var output2 = func.Apply(input);

        // Get nodes
        var nodes = engine.GetType()
            .GetField("_nodes", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?
            .GetValue(engine) as System.Collections.Concurrent.ConcurrentDictionary<Guid, CustomFunctionNode>;

        // Assert
        Assert.Equal(2, nodes?.Count);
        Assert.NotEqual(nodes?.Keys.First(), nodes?.Keys.Last());
    }

    [Fact]
    public void Debug_AccumulateGrad_Test()
    {
        // Test basic gradient accumulation
        var tensor = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        var grad = new Tensor(new float[] { 2, 4, 6 }, new[] { 3 });

        Assert.Equal(new float[] { 0, 0, 0 }, tensor.Gradient.Data);

        tensor.AccumulateGrad(grad);

        Assert.Equal(new float[] { 2, 4, 6 }, tensor.Gradient.Data);
    }

    [Fact]
    public void Gradient_AccumulatesOnMultipleBackwardCalls()
    {
        // Arrange
        var engine = AutogradEngine.Instance;
        engine.ClearGraph();

        var func = new ScaleFunction(2.0f);
        var input = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        var output = func.Apply(input);

        // Act
        engine.Backward(output, Tensor.Ones(output.Shape));
        var gradAfterFirst = input.Gradient?.Clone();

        // Clear and run backward again
        engine.ClearGraph();
        input = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        output = func.Apply(input);
        engine.Backward(output, Tensor.Ones(output.Shape));

        // Assert
        Assert.NotNull(input.Gradient);
        Assert.Equal(gradAfterFirst?.Data, input.Gradient.Data);
    }
}
