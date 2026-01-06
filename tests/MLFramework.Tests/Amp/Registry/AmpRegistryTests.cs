using MLFramework.Amp;
using MLFramework.Core;
using Xunit;

namespace MLFramework.Tests.Amp.Registry;

/// <summary>
/// Tests for AmpRegistry class
/// </summary>
public class AmpRegistryTests
{
    private readonly AmpConfig _config;
    private class TestOperation { }
    private class AnotherTestOperation { }

    public AmpRegistryTests()
    {
        _config = AmpConfig.CreateBf16();
    }

    [Fact]
    public void Constructor_WithValidConfig_CreatesRegistry()
    {
        var registry = new AmpRegistry(_config);

        Assert.NotNull(registry);
    }

    [Fact]
    public void Constructor_WithNullConfig_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new AmpRegistry(null!));
    }

    [Fact]
    public void GetConfig_ReturnsProvidedConfig()
    {
        var registry = new AmpRegistry(_config);
        var config = registry.GetConfig();

        Assert.Equal(_config, config);
    }

    [Fact]
    public void RegisterWhitelist_AddsOperation()
    {
        var registry = new AmpRegistry(_config);

        registry.RegisterWhitelist(typeof(TestOperation));

        Assert.True(registry.IsWhitelisted(typeof(TestOperation)));
    }

    [Fact]
    public void RegisterWhitelist_WithNullType_ThrowsArgumentNullException()
    {
        var registry = new AmpRegistry(_config);

        Assert.Throws<ArgumentNullException>(() =>
            registry.RegisterWhitelist(null!));
    }

    [Fact]
    public void RegisterWhitelist_WithPriority_SetsPriority()
    {
        var registry = new AmpRegistry(_config);

        registry.RegisterWhitelist(typeof(TestOperation), 100);
        var rule = registry.GetRule(typeof(TestOperation));

        Assert.Equal(100, rule?.Priority);
    }

    [Fact]
    public void RegisterBlacklist_AddsOperation()
    {
        var registry = new AmpRegistry(_config);

        registry.RegisterBlacklist(typeof(TestOperation));

        Assert.True(registry.IsBlacklisted(typeof(TestOperation)));
    }

    [Fact]
    public void RegisterBlacklist_WithNullType_ThrowsArgumentNullException()
    {
        var registry = new AmpRegistry(_config);

        Assert.Throws<ArgumentNullException>(() =>
            registry.RegisterBlacklist(null!));
    }

    [Fact]
    public void RegisterBlacklist_WithPriority_SetsPriority()
    {
        var registry = new AmpRegistry(_config);

        registry.RegisterBlacklist(typeof(TestOperation), 50);
        var rule = registry.GetRule(typeof(TestOperation));

        Assert.Equal(50, rule?.Priority);
    }

    [Fact]
    public void RegisterCustomOp_AddsOperationWithCustomDtypes()
    {
        var registry = new AmpRegistry(_config);

        registry.RegisterCustomOp(
            typeof(TestOperation),
            DataType.Float16,
            DataType.Float32,
            10
        );

        var rule = registry.GetRule(typeof(TestOperation));
        Assert.NotNull(rule);
        Assert.Equal(DataType.Float16, rule?.CustomForwardDtype);
        Assert.Equal(DataType.Float32, rule?.CustomBackwardDtype);
    }

    [Fact]
    public void RegisterCustomOp_WithNullType_ThrowsArgumentNullException()
    {
        var registry = new AmpRegistry(_config);

        Assert.Throws<ArgumentNullException>(() =>
            registry.RegisterCustomOp(null!, DataType.Float16, DataType.Float32));
    }

    [Fact]
    public void RegisterRule_AddsRule()
    {
        var registry = new AmpRegistry(_config);
        var rule = new OpPrecisionRule(
            typeof(TestOperation),
            OpPrecision.Lower,
            OpPrecision.Keep
        );

        registry.RegisterRule(rule);

        Assert.Equal(rule, registry.GetRule(typeof(TestOperation)));
    }

    [Fact]
    public void RegisterRule_WithNullRule_ThrowsArgumentNullException()
    {
        var registry = new AmpRegistry(_config);

        Assert.Throws<ArgumentNullException>(() =>
            registry.RegisterRule(null!));
    }

    [Fact]
    public void RegisterRule_WithHigherPriority_OverridesExistingRule()
    {
        var registry = new AmpRegistry(_config);

        registry.RegisterWhitelist(typeof(TestOperation), 10);
        registry.RegisterBlacklist(typeof(TestOperation), 20);

        var rule = registry.GetRule(typeof(TestOperation));
        Assert.Equal(OpPrecision.Higher, rule?.ForwardPrecision);
    }

    [Fact]
    public void RegisterRule_WithSamePriority_OverridesExistingRule()
    {
        var registry = new AmpRegistry(_config);

        registry.RegisterWhitelist(typeof(TestOperation), 10);
        registry.RegisterBlacklist(typeof(TestOperation), 10);

        var rule = registry.GetRule(typeof(TestOperation));
        Assert.Equal(OpPrecision.Higher, rule?.ForwardPrecision);
    }

    [Fact]
    public void RegisterRule_WithLowerPriority_DoesNotOverride()
    {
        var registry = new AmpRegistry(_config);

        registry.RegisterWhitelist(typeof(TestOperation), 20);
        registry.RegisterBlacklist(typeof(TestOperation), 10);

        var rule = registry.GetRule(typeof(TestOperation));
        Assert.Equal(OpPrecision.Lower, rule?.ForwardPrecision);
    }

    [Fact]
    public void GetRule_ReturnsCorrectRule()
    {
        var registry = new AmpRegistry(_config);
        registry.RegisterWhitelist(typeof(TestOperation));

        var rule = registry.GetRule(typeof(TestOperation));

        Assert.NotNull(rule);
        Assert.Equal(typeof(TestOperation), rule?.OperationType);
    }

    [Fact]
    public void GetRule_WithNonExistentOperation_ReturnsNull()
    {
        var registry = new AmpRegistry(_config);

        var rule = registry.GetRule(typeof(TestOperation));

        Assert.Null(rule);
    }

    [Fact]
    public void GetRule_WithNullType_ThrowsArgumentNullException()
    {
        var registry = new AmpRegistry(_config);

        Assert.Throws<ArgumentNullException>(() =>
            registry.GetRule(null!));
    }

    [Fact]
    public void GetForwardDtype_Whitelisted_ReturnsTargetPrecision()
    {
        var registry = new AmpRegistry(_config);
        registry.RegisterWhitelist(typeof(TestOperation));

        var dtype = registry.GetForwardDtype(typeof(TestOperation), DataType.Float32);

        Assert.Equal(DataType.BFloat16, dtype);
    }

    [Fact]
    public void GetForwardDtype_Blacklisted_ReturnsHigherPrecision()
    {
        var registry = new AmpRegistry(_config);
        registry.RegisterBlacklist(typeof(TestOperation));

        var dtype = registry.GetForwardDtype(typeof(TestOperation), DataType.Float32);

        Assert.Equal(DataType.Float32, dtype);
    }

    [Fact]
    public void GetForwardDtype_NoRule_ReturnsInputDtype()
    {
        var registry = new AmpRegistry(_config);

        var dtype = registry.GetForwardDtype(typeof(TestOperation), DataType.Float16);

        Assert.Equal(DataType.Float16, dtype);
    }

    [Fact]
    public void GetBackwardDtype_Whitelisted_ReturnsTargetPrecision()
    {
        var registry = new AmpRegistry(_config);
        registry.RegisterWhitelist(typeof(TestOperation));

        var dtype = registry.GetBackwardDtype(typeof(TestOperation), DataType.Float32);

        Assert.Equal(DataType.BFloat16, dtype);
    }

    [Fact]
    public void GetBackwardDtype_Blacklisted_ReturnsHigherPrecision()
    {
        var registry = new AmpRegistry(_config);
        registry.RegisterBlacklist(typeof(TestOperation));

        var dtype = registry.GetBackwardDtype(typeof(TestOperation), DataType.Float32);

        Assert.Equal(DataType.Float32, dtype);
    }

    [Fact]
    public void GetBackwardDtype_NoRule_ReturnsInputDtype()
    {
        var registry = new AmpRegistry(_config);

        var dtype = registry.GetBackwardDtype(typeof(TestOperation), DataType.Float16);

        Assert.Equal(DataType.Float16, dtype);
    }

    [Fact]
    public void IsWhitelisted_ReturnsCorrectValue()
    {
        var registry = new AmpRegistry(_config);
        registry.RegisterWhitelist(typeof(TestOperation));
        registry.RegisterBlacklist(typeof(AnotherTestOperation));

        Assert.True(registry.IsWhitelisted(typeof(TestOperation)));
        Assert.False(registry.IsWhitelisted(typeof(AnotherTestOperation)));
    }

    [Fact]
    public void IsWhitelisted_WithNullType_ThrowsArgumentNullException()
    {
        var registry = new AmpRegistry(_config);

        Assert.Throws<ArgumentNullException>(() =>
            registry.IsWhitelisted(null!));
    }

    [Fact]
    public void IsBlacklisted_ReturnsCorrectValue()
    {
        var registry = new AmpRegistry(_config);
        registry.RegisterWhitelist(typeof(TestOperation));
        registry.RegisterBlacklist(typeof(AnotherTestOperation));

        Assert.False(registry.IsBlacklisted(typeof(TestOperation)));
        Assert.True(registry.IsBlacklisted(typeof(AnotherTestOperation)));
    }

    [Fact]
    public void IsBlacklisted_WithNullType_ThrowsArgumentNullException()
    {
        var registry = new AmpRegistry(_config);

        Assert.Throws<ArgumentNullException>(() =>
            registry.IsBlacklisted(null!));
    }

    [Fact]
    public void Unregister_RemovesRule()
    {
        var registry = new AmpRegistry(_config);
        registry.RegisterWhitelist(typeof(TestOperation));

        Assert.True(registry.IsWhitelisted(typeof(TestOperation)));

        registry.Unregister(typeof(TestOperation));

        Assert.False(registry.IsWhitelisted(typeof(TestOperation)));
    }

    [Fact]
    public void Unregister_WithNullType_ThrowsArgumentNullException()
    {
        var registry = new AmpRegistry(_config);

        Assert.Throws<ArgumentNullException>(() =>
            registry.Unregister(null!));
    }

    [Fact]
    public void Clear_RemovesAllRules()
    {
        var registry = new AmpRegistry(_config);
        registry.RegisterWhitelist(typeof(TestOperation));
        registry.RegisterBlacklist(typeof(AnotherTestOperation));

        registry.Clear();

        Assert.Null(registry.GetRule(typeof(TestOperation)));
        Assert.Null(registry.GetRule(typeof(AnotherTestOperation)));
    }

    [Fact]
    public void GetAllRules_ReturnsAllRegisteredRules()
    {
        var registry = new AmpRegistry(_config);
        registry.RegisterWhitelist(typeof(TestOperation));
        registry.RegisterBlacklist(typeof(AnotherTestOperation));

        var rules = registry.GetAllRules();

        Assert.Equal(2, rules.Count);
        Assert.True(rules.ContainsKey(typeof(TestOperation)));
        Assert.True(rules.ContainsKey(typeof(AnotherTestOperation)));
    }

    [Fact]
    public void GetAllRules_ReturnsReadOnlyCopy()
    {
        var registry = new AmpRegistry(_config);
        registry.RegisterWhitelist(typeof(TestOperation));

        var rules1 = registry.GetAllRules();
        registry.RegisterBlacklist(typeof(AnotherTestOperation));

        var rules2 = registry.GetAllRules();

        Assert.NotEqual(rules1.Count, rules2.Count);
    }
}
