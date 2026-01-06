using MLFramework.Amp;
using MLFramework.Core;
using MLFramework.Optimizers.MixedPrecision;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Amp.Integration;

/// <summary>
/// End-to-end integration tests for AMP functionality
/// </summary>
public class AmpIntegrationTests
{
    private class TestConvolution { }
    private class TestSoftmax { }

    [Fact]
    public void TrainingLoop_BasicAMP_WorksCorrectly()
    {
        // Setup
        var config = AmpConfig.CreateBf16();
        var registry = new AmpRegistry(config);
        var options = MixedPrecisionOptions.ForBF16();
        var scaler = new DynamicLossScaler(options);

        // Configure some operations
        registry.RegisterWhitelist(typeof(TestConvolution));
        registry.RegisterBlacklist(typeof(TestSoftmax));

        // Verify setup
        Assert.NotNull(registry);
        Assert.NotNull(scaler);
        Assert.True(scaler.IsEnabled);
    }

    [Fact]
    public void TrainingLoop_DynamicScaler_AdjustsScale()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.InitialLossScale = 1000.0f;
        options.GrowthInterval = 2;
        options.GrowthFactor = 2.0f;

        var scaler = new DynamicLossScaler(options);
        var initialScale = scaler.CurrentScale;

        // Simulate successful steps
        scaler.UpdateScale(hadOverflow: false);
        scaler.UpdateScale(hadOverflow: false);

        // Scale should have grown
        Assert.Equal(initialScale * 2.0f, scaler.CurrentScale);
    }

    [Fact]
    public void TrainingLoop_OverflowHandling_SkipsCorrectly()
    {
        var options = MixedPrecisionOptions.ForFP16();
        var scaler = new DynamicLossScaler(options);
        var initialScale = scaler.CurrentScale;

        // Simulate overflow
        var shouldSkip = scaler.UpdateScale(hadOverflow: true);

        // Step should be skipped
        Assert.True(shouldSkip);

        // Scale should have decreased
        Assert.True(scaler.CurrentScale < initialScale);

        // Overflow counter should have increased
        Assert.Equal(1, scaler.ConsecutiveOverflows);
    }

    [Fact]
    public void AutoCast_WithRegistry_SelectsCorrectDtype()
    {
        var config = AmpConfig.CreateFp16();
        var registry = new AmpRegistry(config);
        registry.RegisterWhitelist(typeof(TestConvolution));
        registry.RegisterBlacklist(typeof(TestSoftmax));

        var autoCast = new AutoCast(AutoCastMode.Fp16, true, registry);

        // Convolution should use FP16
        var convDtype = autoCast.GetForwardDtype(typeof(TestConvolution), DataType.Float32);
        Assert.Equal(DataType.Float16, convDtype);

        // Softmax should use FP32
        var softmaxDtype = autoCast.GetForwardDtype(typeof(TestSoftmax), DataType.Float32);
        Assert.Equal(DataType.Float32, softmaxDtype);
    }

    [Fact]
    public void AmpRegistry_WithMultipleConfigurations_WorksCorrectly()
    {
        var fp16Config = AmpConfig.CreateFp16();
        var bf16Config = AmpConfig.CreateBf16();

        var fp16Registry = new AmpRegistry(fp16Config);
        var bf16Registry = new AmpRegistry(bf16Config);

        // Register same operation in both registries
        fp16Registry.RegisterWhitelist(typeof(TestConvolution));
        bf16Registry.RegisterWhitelist(typeof(TestConvolution));

        // Get forward dtypes
        var fp16Dtype = fp16Registry.GetForwardDtype(typeof(TestConvolution), DataType.Float32);
        var bf16Dtype = bf16Registry.GetForwardDtype(typeof(TestConvolution), DataType.Float32);

        // Should match their respective target precisions
        Assert.Equal(DataType.Float16, fp16Dtype);
        Assert.Equal(DataType.BFloat16, bf16Dtype);
    }

    [Fact]
    public void MixedPrecisionOptions_FactoryMethods_CreateCorrectConfigs()
    {
        var fp16Options = MixedPrecisionOptions.ForFP16();
        var bf16Options = MixedPrecisionOptions.ForBF16();
        var conservativeOptions = MixedPrecisionOptions.Conservative();

        // Verify FP16 options
        Assert.Equal(Precision.FP16, fp16Options.Precision);
        Assert.Equal(65536.0f, fp16Options.InitialLossScale);

        // Verify BF16 options
        Assert.Equal(Precision.BF16, bf16Options.Precision);
        Assert.Equal(1.0f, bf16Options.InitialLossScale);

        // Verify Conservative options
        Assert.Equal(Precision.FP16, conservativeOptions.Precision);
        Assert.Equal(8192.0f, conservativeOptions.InitialLossScale);
    }

    [Fact]
    public void DynamicLossScaler_MultipleOverflows_HandlesCorrectly()
    {
        var options = MixedPrecisionOptions.ForFP16();
        options.InitialLossScale = 1000.0f;
        options.BackoffFactor = 0.5f;
        options.MinLossScale = 10.0f;

        var scaler = new DynamicLossScaler(options);

        // Trigger multiple overflows
        for (int i = 0; i < 5; i++)
        {
            scaler.UpdateScale(hadOverflow: true);
        }

        // Should have decreased but not below min
        Assert.True(scaler.CurrentScale >= options.MinLossScale);
        Assert.Equal(5, scaler.ConsecutiveOverflows);
        Assert.Equal(5, scaler.TotalOverflows);
    }

    [Fact]
    public void AmpConfig_DifferentModes_WorkCorrectly()
    {
        var fp16Config = AmpConfig.CreateFp16();
        var bf16Config = AmpConfig.CreateBf16();
        var defaultConfig = AmpConfig.CreateDefault();

        // Verify target precisions
        Assert.Equal(DataType.Float16, fp16Config.TargetPrecision);
        Assert.Equal(DataType.BFloat16, bf16Config.TargetPrecision);
        Assert.Equal(DataType.BFloat16, defaultConfig.TargetPrecision);

        // Verify all are enabled
        Assert.True(fp16Config.Enabled);
        Assert.True(bf16Config.Enabled);
        Assert.True(defaultConfig.Enabled);
    }

    [Fact]
    public void DataTypeExtensions_AmpRelatedMethods_WorkCorrectly()
    {
        // Test size methods
        Assert.Equal(2, DataType.Float16.GetSize());
        Assert.Equal(2, DataType.BFloat16.GetSize());
        Assert.Equal(4, DataType.Float32.GetSize());

        // Test type checking
        Assert.True(DataType.Float16.IsFloatType());
        Assert.True(DataType.BFloat16.IsFloatType());
        Assert.True(DataType.Float16.IsLowPrecision());
        Assert.True(DataType.BFloat16.IsLowPrecision());
        Assert.False(DataType.Float32.IsLowPrecision());

        // Test precision hierarchy
        Assert.Equal(DataType.Float32, DataType.Float16.GetHigherPrecision());
        Assert.Equal(DataType.BFloat16, DataType.Float32.GetLowerPrecision());
    }

    [Fact]
    public void OpPrecisionRule_CustomPrecisions_WorkCorrectly()
    {
        var config = AmpConfig.CreateFp16();
        var rule = new MLFramework.Amp.OpPrecisionRule(
            typeof(TestConvolution),
            OpPrecision.Custom,
            OpPrecision.Custom
        )
        {
            CustomForwardDtype = DataType.BFloat16,
            CustomBackwardDtype = DataType.Float64
        };

        var forwardDtype = rule.GetForwardDtype(config);
        var backwardDtype = rule.GetBackwardDtype(config);

        Assert.Equal(DataType.BFloat16, forwardDtype);
        Assert.Equal(DataType.Float64, backwardDtype);
    }

    [Fact]
    public void CompleteWorkflow_ConfigRegistryScaler_WorkTogether()
    {
        // Step 1: Create AMP configuration
        var config = AmpConfig.CreateFp16();
        Assert.Equal(DataType.Float16, config.TargetPrecision);

        // Step 2: Create registry with configuration
        var registry = new AmpRegistry(config);
        Assert.NotNull(registry);

        // Step 3: Apply default rules
        DefaultAmpRules.ApplyDefaultRules(registry);
        Assert.True(registry.GetAllRules().Count > 0);

        // Step 4: Create loss scaler
        var options = MixedPrecisionOptions.ForFP16();
        var scaler = new DynamicLossScaler(options);
        Assert.True(scaler.IsEnabled);

        // Step 5: Create AutoCast context
        var autoCast = new AutoCast(AutoCastMode.Fp16, true, registry);
        Assert.True(autoCast.Enabled);

        // Step 6: Verify everything works together
        var initialScale = scaler.CurrentScale;
        scaler.UpdateScale(hadOverflow: false);
        Assert.Equal(initialScale, scaler.CurrentScale); // Should be same after 1 step

        var dtype = autoCast.GetForwardDtype(typeof(TestConvolution), DataType.Float32);
        Assert.NotNull(dtype);
    }
}
