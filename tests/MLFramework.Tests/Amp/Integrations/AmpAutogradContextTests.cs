using System;
using Xunit;
using MLFramework.Amp.Integrations;
using MLFramework.Amp;
using MLFramework.Core;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Amp.Integrations
{
    public class AmpAutogradContextTests
    {
        [Fact]
        public void Constructor_WithDefaultValues_CreatesValidContext()
        {
            var context = new AmpAutogradContext();

            Assert.Equal(AutoCastMode.Bf16, context.Mode);
            Assert.Null(context.Registry);
            Assert.Null(context.LossScaler);
            Assert.False(context.NeedsGradientUnscaling);
        }

        [Fact]
        public void Constructor_WithCustomValues_CreatesValidContext()
        {
            var config = AmpConfig.CreateFp16();
            var registry = new AmpRegistry(config);
            var scaler = new StaticLossScaler(1.0f);

            var context = new AmpAutogradContext(
                AutoCastMode.Fp16,
                registry,
                scaler,
                true);

            Assert.Equal(AutoCastMode.Fp16, context.Mode);
            Assert.Equal(registry, context.Registry);
            Assert.Equal(scaler, context.LossScaler);
            Assert.True(context.NeedsGradientUnscaling);
        }

        [Fact]
        public void Properties_CanBeSet()
        {
            var context = new AmpAutogradContext();
            var config = AmpConfig.CreateBf16();
            var registry = new AmpRegistry(config);

            context.Mode = AutoCastMode.None;
            context.Registry = registry;
            context.NeedsGradientUnscaling = true;

            Assert.Equal(AutoCastMode.None, context.Mode);
            Assert.Equal(registry, context.Registry);
            Assert.True(context.NeedsGradientUnscaling);
        }
    }
}
