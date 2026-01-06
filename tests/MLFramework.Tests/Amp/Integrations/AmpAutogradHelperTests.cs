using System;
using Xunit;
using MLFramework.Amp.Integrations;
using MLFramework.Amp;
using MLFramework.Core;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Amp.Integrations
{
    public class AmpAutogradHelperTests
    {
        [Fact]
        public void ConvertGradientsDtype_AllSameDtype_ReturnsOriginalGradients()
        {
            var gradients = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float32) },
                { "param2", CreateTensor(DataType.Float32) }
            };

            var result = AmpAutogradHelper.ConvertGradientsDtype(gradients, DataType.Float32);

            Assert.Equal(2, result.Count);
            Assert.True(result["param1"].IsDtype(DataType.Float32));
            Assert.True(result["param2"].IsDtype(DataType.Float32));
        }

        [Fact]
        public void ConvertGradientsDtype_MixedDtypes_ConvertsAll()
        {
            var gradients = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float16) },
                { "param2", CreateTensor(DataType.BFloat16) }
            };

            var result = AmpAutogradHelper.ConvertGradientsDtype(gradients, DataType.Float32);

            Assert.Equal(2, result.Count);
            Assert.True(result["param1"].IsDtype(DataType.Float32));
            Assert.True(result["param2"].IsDtype(DataType.Float32));
        }

        [Fact]
        public void ConvertGradientsDtype_EmptyDictionary_ReturnsEmpty()
        {
            var gradients = new Dictionary<string, Tensor>();

            var result = AmpAutogradHelper.ConvertGradientsDtype(gradients, DataType.Float32);

            Assert.Empty(result);
        }

        [Fact]
        public void PrepareGradientsForOptimizer_WithoutOverflow_ReturnsTrue()
        {
            var gradients = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float32) }
            };
            var scaler = new StaticLossScaler(1.0f);

            var result = AmpAutogradHelper.PrepareGradientsForOptimizer(gradients, scaler, checkOverflow: false);

            Assert.True(result);
        }

        [Fact]
        public void PrepareGradientsForOptimizer_WithOverflow_ReturnsFalse()
        {
            var gradients = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float32) }
            };

            // Create a scaler that will detect overflow
            var config = new DynamicScalerConfig(
                initScale: 1.0f,
                growthFactor: 2.0f,
                backoffFactor: 0.5f,
                growthInterval: 2000,
                maxScale: 65536.0f);
            var scaler = new DynamicLossScaler(config);
            scaler.UpdateScale(true); // Simulate overflow

            var result = AmpAutogradHelper.PrepareGradientsForOptimizer(gradients, scaler, checkOverflow: true);

            Assert.False(result);
        }

        [Fact]
        public void PrepareGradientsForOptimizer_ConvertsGradientsToFloat32()
        {
            var gradients = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float16) }
            };
            var scaler = new StaticLossScaler(1.0f);

            AmpAutogradHelper.PrepareGradientsForOptimizer(gradients, scaler, checkOverflow: false);

            Assert.True(gradients["param1"].IsDtype(DataType.Float32));
        }

        [Fact]
        public void EnsureGradientCompatibility_CompatibleParameters_ReturnsTrue()
        {
            var parameters = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float32, new[] { 2, 2 }) },
                { "param2", CreateTensor(DataType.Float32, new[] { 3, 3 }) }
            };

            var gradients = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float32, new[] { 2, 2 }) },
                { "param2", CreateTensor(DataType.Float32, new[] { 3, 3 }) }
            };

            var result = AmpAutogradHelper.EnsureGradientCompatibility(parameters, gradients);

            Assert.True(result);
        }

        [Fact]
        public void EnsureGradientCompatibility_ShapeMismatch_ReturnsFalse()
        {
            var parameters = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float32, new[] { 2, 2 }) }
            };

            var gradients = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float32, new[] { 3, 3 }) }
            };

            var result = AmpAutogradHelper.EnsureGradientCompatibility(parameters, gradients);

            Assert.False(result);
        }

        [Fact]
        public void EnsureGradientCompatibility_MissingGradient_ReturnsFalse()
        {
            var parameters = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float32) },
                { "param2", CreateTensor(DataType.Float32) }
            };

            var gradients = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float32) }
            };

            var result = AmpAutogradHelper.EnsureGradientCompatibility(parameters, gradients);

            Assert.False(result);
        }

        [Fact]
        public void EnsureGradientCompatibility_DtypeMismatch_ReturnsFalse()
        {
            var parameters = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float32) }
            };

            var gradients = new Dictionary<string, Tensor>
            {
                { "param1", CreateTensor(DataType.Float16) }
            };

            var result = AmpAutogradHelper.EnsureGradientCompatibility(parameters, gradients);

            Assert.False(result);
        }

        [Fact]
        public void CreateAmpGraph_ReturnsValidGraph()
        {
            var tensor = CreateTensor(DataType.Float32);

            var graph = AmpAutogradHelper.CreateAmpGraph(tensor, AutoCastMode.Bf16);

            Assert.NotNull(graph);
        }

        [Fact]
        public void PrepareGradientsForOptimizer_NullGradients_ThrowsArgumentNullException()
        {
            Dictionary<string, Tensor>? gradients = null;
            var scaler = new StaticLossScaler(1.0f);

            Assert.Throws<ArgumentNullException>(() =>
                AmpAutogradHelper.PrepareGradientsForOptimizer(gradients!, scaler));
        }

        [Fact]
        public void EnsureGradientCompatibility_NullParameters_ThrowsArgumentNullException()
        {
            var gradients = new Dictionary<string, Tensor>();

            Assert.Throws<ArgumentNullException>(() =>
                AmpAutogradHelper.EnsureGradientCompatibility(null!, gradients));
        }

        private Tensor CreateTensor(DataType dtype, int[]? shape = null)
        {
            var s = shape ?? new[] { 2 };
            var data = new float[2];
            var tensor = new Tensor(data, s);
            tensor.SetDtype(dtype);
            return tensor;
        }
    }
}
