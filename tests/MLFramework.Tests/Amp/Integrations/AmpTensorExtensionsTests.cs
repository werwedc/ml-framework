using System;
using Xunit;
using MLFramework.Amp.Integrations;
using MLFramework.Amp;
using MLFramework.Core;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Amp.Integrations
{
    public class AmpTensorExtensionsTests
    {
        [Fact]
        public void BackwardAmp_WithoutLossScaler_ComputesGradients()
        {
            var data = new float[] { 1.0f, 2.0f };
            var shape = new[] { 2 };
            var tensor = new Tensor(data, shape, requiresGrad: true);

            // Should not throw
            tensor.BackwardAmp();
        }

        [Fact]
        public void BackwardAmp_WithLossScaler_ComputesGradients()
        {
            var data = new float[] { 1.0f, 2.0f };
            var shape = new[] { 2 };
            var tensor = new Tensor(data, shape, requiresGrad: true);
            var scaler = new StaticLossScaler(1.0f);

            // Should not throw
            tensor.BackwardAmp(scaler);
        }

        [Fact]
        public void BackwardAmp_WithRetainGraph_ComputesGradients()
        {
            var data = new float[] { 1.0f, 2.0f };
            var shape = new[] { 2 };
            var tensor = new Tensor(data, shape, requiresGrad: true);

            // Should not throw
            tensor.BackwardAmp(retainGraph: true);
        }

        [Fact]
        public void GradAmp_WithoutGradients_ThrowsInvalidOperationException()
        {
            var data = new float[] { 1.0f, 2.0f };
            var shape = new[] { 2 };
            var tensor = new Tensor(data, shape, requiresGrad: true);

            Assert.Throws<InvalidOperationException>(() => tensor.GradAmp());
        }

        [Fact]
        public void GradAmp_WithGradients_ReturnsGradient()
        {
            var data = new float[] { 1.0f, 2.0f };
            var shape = new[] { 1, 2 };
            var tensor = new Tensor(data, shape, requiresGrad: true);

            tensor.Backward();

            var grad = tensor.GradAmp();

            Assert.NotNull(grad);
            Assert.Equal(DataType.Float32, grad.GetDtype());
        }

        [Fact]
        public void NeedsAmpBackward_WithFloat32Tensor_ReturnsFalse()
        {
            var data = new float[] { 1.0f, 2.0f };
            var shape = new[] { 2 };
            var tensor = new Tensor(data, shape);
            tensor.SetDtype(DataType.Float32);

            Assert.False(tensor.NeedsAmpBackward());
        }

        [Fact]
        public void NeedsAmpBackward_WithFloat16Tensor_ReturnsTrue()
        {
            var data = new float[] { 1.0f, 2.0f };
            var shape = new[] { 2 };
            var tensor = new Tensor(data, shape);
            tensor.SetDtype(DataType.Float16);

            Assert.True(tensor.NeedsAmpBackward());
        }

        [Fact]
        public void NeedsAmpBackward_WithBFloat16Tensor_ReturnsTrue()
        {
            var data = new float[] { 1.0f, 2.0f };
            var shape = new[] { 2 };
            var tensor = new Tensor(data, shape);
            tensor.SetDtype(DataType.BFloat16);

            Assert.True(tensor.NeedsAmpBackward());
        }

        [Fact]
        public void BackwardAmp_NullTensor_ThrowsArgumentNullException()
        {
            Tensor? tensor = null;

            Assert.Throws<ArgumentNullException>(() => tensor!.BackwardAmp());
        }
    }
}
