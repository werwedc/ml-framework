using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Unit tests for custom autograd functions.
/// </summary>
public class AutogradFunctionTests : IDisposable
{
    private readonly GraphBuilder _graphBuilder;

    public AutogradFunctionTests()
    {
        _graphBuilder = new GraphBuilder();
    }

    public void Dispose()
    {
        _graphBuilder.Dispose();
        FunctionRegistry.ClearAll();
    }

    #region Basic Custom Function Tests

    [Fact]
    public void Test_BasicCustomFunction_SquareOperation()
    {
        // Arrange: y = x^2, dy/dx = 2x
        var x = new Tensor(new float[] { 3.0f }, new int[] { 1 }, requiresGrad: true);

        var squareFn = new SquareFunction();
        var y = squareFn.Apply(x);

        // Act
        y.Backward();

        // Assert - dy/dx = 2 * 3 = 6
        Assert.NotNull(x.Gradient);
        Assert.Equal(1, x.Gradient.Size);
        Assert.Equal(6.0f, x.Gradient._data[0], precision: 5);
    }

    [Fact]
    public void Test_CustomFunction_SavesTensor_RetrievesCorrectly()
    {
        // Arrange
        var x = new Tensor(new float[] { 5.0f }, new int[] { 1 }, requiresGrad: true);

        // Act
        var function = new SquareFunction();
        var output = function.Apply(x);

        // Assert - Function should have saved the input tensor
        Assert.Equal(1, function.SavedTensors.Count);
    }

    #endregion

    #region Tensor Saving and Retrieval Tests

    [Fact]
    public void Test_SaveForBackward_SavesMultipleTensors()
    {
        // Arrange
        var function = new MultiInputFunction();
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var y = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // Act
        var output = function.Apply(x, y);

        // Assert
        Assert.Equal(2, function.SavedTensors.Count);
    }

    [Fact]
    public void Test_GetSavedTensor_RetrievesCorrectTensor()
    {
        // Arrange
        var x = new Tensor(new float[] { 3.0f }, new int[] { 1 });
        var function = new SquareFunction();

        // Act
        function.Apply(x);
        var saved = function.GetSavedTensor(0);

        // Assert
        Assert.NotNull(saved);
        Assert.Equal(x._data[0], saved._data[0], precision: 5);
    }

    [Fact]
    public void Test_GetSavedTensor_InvalidIndex_ThrowsException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var function = new SquareFunction();
        function.Apply(x);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => function.GetSavedTensor(5));
    }

    #endregion

    #region Scalar Saving and Retrieval Tests

    [Fact]
    public void Test_SaveScalarForBackward_SavesValue()
    {
        // Arrange
        var function = new ScalarFunction(3.0);
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // Act
        var output = function.Apply(x);

        // Assert
        Assert.Single(function.SavedScalars);
    }

    [Fact]
    public void Test_GetSavedScalar_RetrievesCorrectValue()
    {
        // Arrange
        var function = new ScalarFunction(3.0);
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // Act
        function.Apply(x);
        var saved = function.GetSavedScalar<double>(0);

        // Assert
        Assert.Equal(3.0, saved, precision: 5);
    }

    [Fact]
    public void Test_GetSavedScalar_InvalidType_ThrowsException()
    {
        // Arrange
        var function = new ScalarFunction(3.0);
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // Act
        function.Apply(x);

        // Act & Assert
        Assert.Throws<InvalidCastException>(() => function.GetSavedScalar<string>(0));
    }

    #endregion

    #region Backward Gradient Computation Tests

    [Fact]
    public void Test_Backward_ComputesCorrectGradient()
    {
        // Arrange: y = 2*x, dy/dx = 2
        var x = new Tensor(new float[] { 4.0f }, new int[] { 1 }, requiresGrad: true);
        var function = new MultiplyByTwoFunction();

        // Act
        var y = function.Apply(x);
        y.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(2.0f, x.Gradient._data[0], precision: 5);
    }

    [Fact]
    public void Test_Backward_MultipleInputs_ComputesCorrectGradients()
    {
        // Arrange: z = x * y, dz/dx = y, dz/dy = x
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var y = new Tensor(new float[] { 3.0f }, new int[] { 1 }, requiresGrad: true);
        var function = new MultiplyFunction();

        // Act
        var z = function.Apply(x, y);
        z.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.NotNull(y.Gradient);
        Assert.Equal(3.0f, x.Gradient._data[0], precision: 5); // dz/dx = y = 3
        Assert.Equal(2.0f, y.Gradient._data[0], precision: 5); // dz/dy = x = 2
    }

    #endregion

    #region Graph Integration Tests

    [Fact]
    public void Test_GraphIntegration_CreatesOperationContext()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var function = new SquareFunction();

        // Act
        var output = function.Apply(x);

        // Assert
        Assert.NotNull(function.Context);
        Assert.Equal("Square", function.Context.OperationName);
    }

    [Fact]
    public void Test_GraphIntegration_ContextContainsBackwardFunction()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        var function = new SquareFunction();

        // Act
        var output = function.Apply(x);

        // Assert
        Assert.NotNull(function.Context.BackwardFn);
    }

    #endregion

    #region Function Registry Tests

    [Fact]
    public void Test_FunctionRegistry_Register_Success()
    {
        // Arrange & Act
        FunctionRegistry.Register<SquareFunction>("SquareCustom");

        // Assert
        Assert.True(FunctionRegistry.IsRegistered("SquareCustom"));
    }

    [Fact]
    public void Test_FunctionRegistry_GetFunctionType_ReturnsCorrectType()
    {
        // Arrange
        FunctionRegistry.Register<SquareFunction>("SquareTest");

        // Act
        var functionType = FunctionRegistry.GetFunctionType("SquareTest");

        // Assert
        Assert.Equal(typeof(SquareFunction), functionType);
    }

    [Fact]
    public void Test_FunctionRegistry_Unregister_RemovesFunction()
    {
        // Arrange
        FunctionRegistry.Register<SquareFunction>("SquareToUnregister");

        // Act
        var result = FunctionRegistry.Unregister("SquareToUnregister");

        // Assert
        Assert.True(result);
        Assert.False(FunctionRegistry.IsRegistered("SquareToUnregister"));
    }

    [Fact]
    public void Test_FunctionRegistry_RegisterDuplicate_ThrowsException()
    {
        // Arrange
        FunctionRegistry.Register<SquareFunction>("SquareDup");

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            FunctionRegistry.Register<SquareFunction>("SquareDup"));
    }

    [Fact]
    public void Test_FunctionRegistry_GetNonExistent_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<KeyNotFoundException>(() =>
            FunctionRegistry.GetFunctionType("NonExistent"));
    }

    [Fact]
    public void Test_FunctionRegistry_CreateInstance_ReturnsFunction()
    {
        // Arrange
        FunctionRegistry.Register<SquareFunction>("SquareCreate");

        // Act
        var function = FunctionRegistry.CreateInstance("SquareCreate");

        // Assert
        Assert.NotNull(function);
        Assert.IsType<SquareFunction>(function);
    }

    [Fact]
    public void Test_FunctionRegistry_GetAllRegisteredNames_ReturnsCorrectNames()
    {
        // Arrange
        FunctionRegistry.Register<SquareFunction>("Func1");
        FunctionRegistry.Register<MultiplyFunction>("Func2");

        // Act
        var names = FunctionRegistry.GetAllRegisteredNames();

        // Assert
        Assert.Contains("Func1", names);
        Assert.Contains("Func2", names);
    }

    [Fact]
    public void Test_FunctionRegistry_ClearAll_RemovesAllFunctions()
    {
        // Arrange
        FunctionRegistry.Register<SquareFunction>("Func1");
        FunctionRegistry.Register<MultiplyFunction>("Func2");

        // Act
        FunctionRegistry.ClearAll();

        // Assert
        Assert.Empty(FunctionRegistry.GetAllRegisteredNames());
    }

    #endregion

    #region Extension Methods Tests

    [Fact]
    public void Test_ExtensionMethod_ApplyFunction_ReturnsCorrectResult()
    {
        // Arrange
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);

        // Act
        var y = x.ApplyFunction<SquareFunction>();
        y.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(4.0f, x.Gradient._data[0], precision: 5); // dy/dx = 2*2 = 4
    }

    [Fact]
    public void Test_ExtensionMethod_ApplyFunctionByName_ReturnsCorrectResult()
    {
        // Arrange
        FunctionRegistry.Register<SquareFunction>("SquareExt");
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);

        // Act
        var y = x.ApplyFunction("SquareExt");
        y.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(4.0f, x.Gradient._data[0], precision: 5);
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public void Test_NullInput_ThrowsArgumentNullException()
    {
        // Arrange
        var function = new SquareFunction();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => function.Apply(null!));
    }

    [Fact]
    public void Test_CreateContext_NullName_ThrowsArgumentException()
    {
        // Arrange
        var function = new SquareFunction();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => function.CreateContext(null!));
    }

    [Fact]
    public void Test_FunctionRegistry_RegisterNullName_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            FunctionRegistry.Register<SquareFunction>(null!));
    }

    [Fact]
    public void Test_FunctionRegistry_RegisterInvalidType_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            FunctionRegistry.Register<string>("StringFunc"));
    }

    [Fact]
    public void Test_DisposedFunction_ThrowsObjectDisposedException()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var function = new SquareFunction();
        function.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() => function.Apply(x));
    }

    #endregion

    #region Memory Management Tests

    [Fact]
    public void Test_Dispose_ClearsSavedTensors()
    {
        // Arrange
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var function = new SquareFunction();
        function.Apply(x);

        // Act
        function.Dispose();

        // Assert
        Assert.Empty(function.SavedTensors);
    }

    [Fact]
    public void Test_BackwardPass_ClearsSavedTensors()
    {
        // Arrange
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var function = new SquareFunction();

        // Act
        var y = function.Apply(x);
        Assert.NotEmpty(function.SavedTensors);
        y.Backward();

        // Assert - Saved tensors should be cleared after backward
        Assert.Empty(function.SavedTensors);
    }

    #endregion

    #region Custom Test Functions

    /// <summary>
    /// Test function: y = x^2, dy/dx = 2x
    /// </summary>
    private class SquareFunction : AutogradFunction
    {
        public override Tensor Forward(params Tensor[] inputs)
        {
            var x = inputs[0];
            SaveForBackward(x);

            var result = new Tensor(new float[x.Size], x.Shape, x.RequiresGrad);
            for (int i = 0; i < x.Size; i++)
            {
                result._data[i] = x._data[i] * x._data[i];
            }
            return result;
        }

        public override Tensor[] Backward(Tensor gradOutput)
        {
            var x = GetSavedTensor(0);
            var grad = new Tensor(new float[x.Size], x.Shape);
            for (int i = 0; i < x.Size; i++)
            {
                grad._data[i] = 2.0f * x._data[i] * gradOutput._data[i];
            }
            return new[] { grad };
        }
    }

    /// <summary>
    /// Test function: y = 2*x, dy/dx = 2
    /// </summary>
    private class MultiplyByTwoFunction : AutogradFunction
    {
        public override Tensor Forward(params Tensor[] inputs)
        {
            var x = inputs[0];
            var result = new Tensor(new float[x.Size], x.Shape, x.RequiresGrad);
            for (int i = 0; i < x.Size; i++)
            {
                result._data[i] = 2.0f * x._data[i];
            }
            return result;
        }

        public override Tensor[] Backward(Tensor gradOutput)
        {
            var grad = new Tensor(new float[gradOutput.Size], gradOutput.Shape);
            for (int i = 0; i < gradOutput.Size; i++)
            {
                grad._data[i] = 2.0f * gradOutput._data[i];
            }
            return new[] { grad };
        }
    }

    /// <summary>
    /// Test function: z = x * y, dz/dx = y, dz/dy = x
    /// </summary>
    private class MultiplyFunction : AutogradFunction
    {
        public override Tensor Forward(params Tensor[] inputs)
        {
            var x = inputs[0];
            var y = inputs[1];
            SaveForBackward(x, y);

            var result = new Tensor(new float[x.Size], x.Shape, x.RequiresGrad || y.RequiresGrad);
            for (int i = 0; i < x.Size; i++)
            {
                result._data[i] = x._data[i] * y._data[i];
            }
            return result;
        }

        public override Tensor[] Backward(Tensor gradOutput)
        {
            var x = GetSavedTensor(0);
            var y = GetSavedTensor(1);

            var gradX = new Tensor(new float[x.Size], x.Shape);
            var gradY = new Tensor(new float[y.Size], y.Shape);

            for (int i = 0; i < gradOutput.Size; i++)
            {
                gradX._data[i] = y._data[i] * gradOutput._data[i];
                gradY._data[i] = x._data[i] * gradOutput._data[i];
            }

            return new[] { gradX, gradY };
        }
    }

    /// <summary>
    /// Test function with scalar saving: y = scalar * x
    /// </summary>
    private class ScalarFunction : AutogradFunction
    {
        private readonly double _scalar;

        public ScalarFunction(double scalar)
        {
            _scalar = scalar;
        }

        public override Tensor Forward(params Tensor[] inputs)
        {
            var x = inputs[0];
            SaveScalarForBackward(_scalar);

            var result = new Tensor(new float[x.Size], x.Shape, x.RequiresGrad);
            for (int i = 0; i < x.Size; i++)
            {
                result._data[i] = (float)_scalar * x._data[i];
            }
            return result;
        }

        public override Tensor[] Backward(Tensor gradOutput)
        {
            var scalar = GetSavedScalar<double>(0);
            var grad = new Tensor(new float[gradOutput.Size], gradOutput.Shape);
            for (int i = 0; i < gradOutput.Size; i++)
            {
                grad._data[i] = (float)scalar * gradOutput._data[i];
            }
            return new[] { grad };
        }
    }

    /// <summary>
    /// Test function with multiple inputs
    /// </summary>
    private class MultiInputFunction : AutogradFunction
    {
        public override Tensor Forward(params Tensor[] inputs)
        {
            SaveForBackward(inputs);

            var x = inputs[0];
            var y = inputs[1];
            var result = new Tensor(new float[x.Size], x.Shape);
            for (int i = 0; i < x.Size; i++)
            {
                result._data[i] = x._data[i] + y._data[i];
            }
            return result;
        }

        public override Tensor[] Backward(Tensor gradOutput)
        {
            var grad1 = gradOutput.Clone();
            var grad2 = gradOutput.Clone();
            return new[] { grad1, grad2 };
        }
    }

    #endregion
}
