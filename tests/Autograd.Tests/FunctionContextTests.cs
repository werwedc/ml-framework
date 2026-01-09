using MLFramework.Autograd;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Autograd;

public class FunctionContextTests
{
    [Fact]
    public void SaveForBackward_Tensors_SavesAndRetrievesCorrectly()
    {
        // Arrange
        var ctx = new FunctionContext();
        var tensor1 = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
        var tensor2 = new Tensor(new float[] { 3.0f, 4.0f, 5.0f }, new int[] { 3 });

        // Act
        ctx.SaveForBackward(tensor1, tensor2);

        // Assert
        Assert.Equal(2, ctx.SavedTensorCount);
        Assert.Same(tensor1, ctx.GetSavedTensor(0));
        Assert.Same(tensor2, ctx.GetSavedTensor(1));
    }

    [Fact]
    public void SaveForBackward_Objects_SavesAndRetrievesCorrectly()
    {
        // Arrange
        var ctx = new FunctionContext();

        // Act
        ctx.SaveForBackward(42, "test metadata", 3.14);

        // Assert
        Assert.Equal(3, ctx.SavedObjectCount);
        Assert.Equal(42, ctx.GetSavedObject(0));
        Assert.Equal("test metadata", ctx.GetSavedObject(1));
        Assert.Equal(3.14, ctx.GetSavedObject(2));
    }

    [Fact]
    public void SaveForBackward_EmptyArrays_Succeeds()
    {
        // Arrange
        var ctx = new FunctionContext();

        // Act & Assert
        ctx.SaveForBackward(new Tensor[0]);
        ctx.SaveForBackward(new object[0]);
        Assert.Equal(0, ctx.SavedTensorCount);
        Assert.Equal(0, ctx.SavedObjectCount);
    }

    [Fact]
    public void GetSavedTensor_OutOfRange_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var ctx = new FunctionContext();
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        ctx.SaveForBackward(tensor);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.GetSavedTensor(1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.GetSavedTensor(-1));
    }

    [Fact]
    public void GetSavedObject_OutOfRange_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var ctx = new FunctionContext();
        ctx.SaveForBackward("test");

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.GetSavedObject(1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.GetSavedObject(-1));
    }

    [Fact]
    public void GetSavedTensor_OutOfRange_HasCorrectErrorMessage()
    {
        // Arrange
        var ctx = new FunctionContext();
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        ctx.SaveForBackward(tensor);

        // Act
        var exception = Assert.Throws<ArgumentOutOfRangeException>(() => ctx.GetSavedTensor(5));

        // Assert
        Assert.Contains("Tensor index 5 is out of range", exception.Message);
    }

    [Fact]
    public void GetSavedObject_OutOfRange_HasCorrectErrorMessage()
    {
        // Arrange
        var ctx = new FunctionContext();
        ctx.SaveForBackward("test");

        // Act
        var exception = Assert.Throws<ArgumentOutOfRangeException>(() => ctx.GetSavedObject(5));

        // Assert
        Assert.Contains("Object index 5 is out of range", exception.Message);
    }

    [Fact]
    public void SaveForBackward_NullTensorsArray_TreatedAsEmpty()
    {
        // Arrange
        var ctx = new FunctionContext();

        // Act - null array is treated as empty
        ctx.SaveForBackward((Tensor[])null);

        // Assert
        Assert.Equal(0, ctx.SavedTensorCount);
    }

    [Fact]
    public void SaveForBackward_NullObjectsArray_TreatedAsEmpty()
    {
        // Arrange
        var ctx = new FunctionContext();

        // Act - null array is treated as empty
        ctx.SaveForBackward((object[])null);

        // Assert
        Assert.Equal(0, ctx.SavedObjectCount);
    }

    [Fact]
    public void SaveForBackward_NullTensorValues_SavesSuccessfully()
    {
        // Arrange
        var ctx = new FunctionContext();

        // Act - Explicitly cast to Tensor[] to resolve overload ambiguity
        ctx.SaveForBackward((Tensor)null, (Tensor)null);

        // Assert
        Assert.Equal(2, ctx.SavedTensorCount);
        Assert.Null(ctx.GetSavedTensor(0));
        Assert.Null(ctx.GetSavedTensor(1));
    }

    [Fact]
    public void SaveForBackward_NullObjectValues_SavesSuccessfully()
    {
        // Arrange
        var ctx = new FunctionContext();

        // Act - Explicitly cast to object to resolve overload ambiguity
        ctx.SaveForBackward((object)null, (object)null);

        // Assert
        Assert.Equal(2, ctx.SavedObjectCount);
        Assert.Null(ctx.GetSavedObject(0));
        Assert.Null(ctx.GetSavedObject(1));
    }

    [Fact]
    public void GetSavedTensor_FromEmptyContext_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var ctx = new FunctionContext();

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.GetSavedTensor(0));
    }

    [Fact]
    public void GetSavedObject_FromEmptyContext_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var ctx = new FunctionContext();

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => ctx.GetSavedObject(0));
    }

    [Fact]
    public void Clear_ClearsSavedTensorsAndObjects()
    {
        // Arrange
        var ctx = new FunctionContext();
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        ctx.SaveForBackward(tensor);
        ctx.SaveForBackward(42, "test");

        // Act
        ctx.Clear();

        // Assert
        Assert.Equal(0, ctx.SavedTensorCount);
        Assert.Equal(0, ctx.SavedObjectCount);
    }

    [Fact]
    public void Dispose_ClearsSavedTensorsAndObjects()
    {
        // Arrange
        var ctx = new FunctionContext();
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        ctx.SaveForBackward(tensor);
        ctx.SaveForBackward(42, "test");

        // Act
        ctx.Dispose();

        // Assert
        Assert.Equal(0, ctx.SavedTensorCount);
        Assert.Equal(0, ctx.SavedObjectCount);
        Assert.True(ctx.IsDisposed);
    }

    [Fact]
    public void Dispose_SetsIsDisposedToTrue()
    {
        // Arrange
        var ctx = new FunctionContext();

        // Act
        ctx.Dispose();

        // Assert
        Assert.True(ctx.IsDisposed);
    }

    [Fact]
    public void SaveForBackward_AfterDispose_ThrowsInvalidOperationException()
    {
        // Arrange
        var ctx = new FunctionContext();
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        ctx.Dispose();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => ctx.SaveForBackward(tensor));
    }

    [Fact]
    public void SaveForBackward_Objects_AfterDispose_ThrowsInvalidOperationException()
    {
        // Arrange
        var ctx = new FunctionContext();
        ctx.Dispose();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => ctx.SaveForBackward(42));
    }

    [Fact]
    public void GetSavedTensor_AfterDispose_ThrowsInvalidOperationException()
    {
        // Arrange
        var ctx = new FunctionContext();
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        ctx.SaveForBackward(tensor);
        ctx.Dispose();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => ctx.GetSavedTensor(0));
    }

    [Fact]
    public void GetSavedObject_AfterDispose_ThrowsInvalidOperationException()
    {
        // Arrange
        var ctx = new FunctionContext();
        ctx.SaveForBackward(42);
        ctx.Dispose();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => ctx.GetSavedObject(0));
    }

    [Fact]
    public void GetSavedTensor_AfterDispose_HasCorrectErrorMessage()
    {
        // Arrange
        var ctx = new FunctionContext();
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        ctx.SaveForBackward(tensor);
        ctx.Dispose();

        // Act
        var exception = Assert.Throws<InvalidOperationException>(() => ctx.GetSavedTensor(0));

        // Assert
        Assert.Contains("disposed context", exception.Message);
    }

    [Fact]
    public void GetSavedObject_AfterDispose_HasCorrectErrorMessage()
    {
        // Arrange
        var ctx = new FunctionContext();
        ctx.SaveForBackward(42);
        ctx.Dispose();

        // Act
        var exception = Assert.Throws<InvalidOperationException>(() => ctx.GetSavedObject(0));

        // Assert
        Assert.Contains("disposed context", exception.Message);
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Arrange
        var ctx = new FunctionContext();

        // Act & Assert (should not throw)
        ctx.Dispose();
        ctx.Dispose();
        Assert.True(ctx.IsDisposed);
    }

    [Fact]
    public void Clear_CanBeCalledMultipleTimes()
    {
        // Arrange
        var ctx = new FunctionContext();
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        ctx.SaveForBackward(tensor);

        // Act
        ctx.Clear();
        ctx.Clear();

        // Assert
        Assert.Equal(0, ctx.SavedTensorCount);
    }

    [Fact]
    public void MultipleInvocations_EachContextIsIndependent()
    {
        // Arrange
        var ctx1 = new FunctionContext();
        var ctx2 = new FunctionContext();
        var tensor1 = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var tensor2 = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // Act
        ctx1.SaveForBackward(tensor1);
        ctx2.SaveForBackward(tensor2);

        // Assert
        Assert.Equal(1, ctx1.SavedTensorCount);
        Assert.Equal(1, ctx2.SavedTensorCount);
        Assert.Same(tensor1, ctx1.GetSavedTensor(0));
        Assert.Same(tensor2, ctx2.GetSavedTensor(0));
    }

    [Fact]
    public void SaveForBackward_MixedTensorsAndObjects_MaintainsSeparateLists()
    {
        // Arrange
        var ctx = new FunctionContext();
        var tensor1 = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var tensor2 = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // Act
        ctx.SaveForBackward(tensor1, tensor2);
        ctx.SaveForBackward(42, "test", 3.14);

        // Assert
        Assert.Equal(2, ctx.SavedTensorCount);
        Assert.Equal(3, ctx.SavedObjectCount);
        Assert.Same(tensor1, ctx.GetSavedTensor(0));
        Assert.Same(tensor2, ctx.GetSavedTensor(1));
        Assert.Equal(42, ctx.GetSavedObject(0));
        Assert.Equal("test", ctx.GetSavedObject(1));
        Assert.Equal(3.14, ctx.GetSavedObject(2));
    }
}
