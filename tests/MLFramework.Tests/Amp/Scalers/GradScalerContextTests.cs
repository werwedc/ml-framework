using MLFramework.Amp;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Amp.Scalers;

/// <summary>
/// Tests for GradScalerContext class
/// </summary>
public class GradScalerContextTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidParameters_CreatesContext()
    {
        var scaler = new GradScaler(scale: 1000.0f);
        var loss = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        var context = new GradScalerContext(scaler, loss);

        Assert.NotNull(context);
        Assert.NotNull(context.ScaledLoss);
        Assert.Equal(1000.0f, context.ScaledLoss[0]);
    }

    [Fact]
    public void Constructor_WithNullScaler_ThrowsArgumentNullException()
    {
        var loss = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        Assert.Throws<ArgumentNullException>(() =>
            new GradScalerContext(null!, loss));
    }

    [Fact]
    public void Constructor_WithNullLoss_ThrowsArgumentNullException()
    {
        var scaler = new GradScaler();

        Assert.Throws<ArgumentNullException>(() =>
            new GradScalerContext(scaler, null!));
    }

    #endregion

    #region PrepareStep Method Tests

    [Fact]
    public void PrepareStep_WithValidGradients_ReturnsTrue()
    {
        var scaler = new GradScaler();
        var loss = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f }, new int[] { 1 }) }
        };

        using (var context = new GradScalerContext(scaler, loss))
        {
            bool prepared = context.PrepareStep(gradients);

            Assert.True(prepared);
        }
    }

    [Fact]
    public void PrepareStep_WithOverflow_ReturnsFalse()
    {
        var scaler = new GradScaler();
        var loss = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { float.NaN }, new int[] { 1 }) }
        };

        using (var context = new GradScalerContext(scaler, loss))
        {
            bool prepared = context.PrepareStep(gradients);

            Assert.False(prepared);
        }
    }

    [Fact]
    public void PrepareStep_WhenCalledTwice_ThrowsInvalidOperationException()
    {
        var scaler = new GradScaler();
        var loss = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f }, new int[] { 1 }) }
        };

        using (var context = new GradScalerContext(scaler, loss))
        {
            context.PrepareStep(gradients);

            Assert.Throws<InvalidOperationException>(() =>
                context.PrepareStep(gradients));
        }
    }

    [Fact]
    public void PrepareStep_WhenDisposed_ThrowsObjectDisposedException()
    {
        var scaler = new GradScaler();
        var loss = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f }, new int[] { 1 }) }
        };

        var context = new GradScalerContext(scaler, loss);
        context.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
            context.PrepareStep(gradients));
    }

    #endregion

    #region PrepareAndGetGradients Method Tests

    [Fact]
    public void PrepareAndGetGradients_WithValidGradients_ReturnsUnscaledGradients()
    {
        var scaler = new GradScaler(scale: 1000.0f);
        var loss = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1000.0f }, new int[] { 1 }) }
        };

        using (var context = new GradScalerContext(scaler, loss))
        {
            var unscaledGradients = context.PrepareAndGetGradients(gradients);

            Assert.NotNull(unscaledGradients);
            Assert.Equal(1.0f, unscaledGradients["param1"][0], precision: 4);
        }
    }

    [Fact]
    public void PrepareAndGetGradients_WithOverflow_ReturnsNull()
    {
        var scaler = new GradScaler();
        var loss = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { float.NaN }, new int[] { 1 }) }
        };

        using (var context = new GradScalerContext(scaler, loss))
        {
            var unscaledGradients = context.PrepareAndGetGradients(gradients);

            Assert.Null(unscaledGradients);
        }
    }

    [Fact]
    public void PrepareAndGetGradients_WhenCalledTwice_ThrowsInvalidOperationException()
    {
        var scaler = new GradScaler();
        var loss = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f }, new int[] { 1 }) }
        };

        using (var context = new GradScalerContext(scaler, loss))
        {
            context.PrepareAndGetGradients(gradients);

            Assert.Throws<InvalidOperationException>(() =>
                context.PrepareAndGetGradients(gradients));
        }
    }

    [Fact]
    public void PrepareAndGetGradients_WhenDisposed_ThrowsObjectDisposedException()
    {
        var scaler = new GradScaler();
        var loss = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var gradients = new Dictionary<string, Tensor>
        {
            { "param1", new Tensor(new float[] { 1.0f }, new int[] { 1 }) }
        };

        var context = new GradScalerContext(scaler, loss);
        context.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
            context.PrepareAndGetGradients(gradients));
    }

    #endregion

    #region Dispose Method Tests

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        var scaler = new GradScaler();
        var loss = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        var context = new GradScalerContext(scaler, loss);
        context.Dispose();
        context.Dispose(); // Should not throw
    }

    #endregion
}
