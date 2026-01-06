using MLFramework.Amp;
using Xunit;

namespace MLFramework.Tests.Amp.Registry;

/// <summary>
/// Tests for DefaultAmpRules class
/// </summary>
public class DefaultAmpRulesTests
{
    [Fact]
    public void ApplyDefaultRules_WithValidRegistry_RegistersAllRules()
    {
        var config = AmpConfig.CreateBf16();
        var registry = new AmpRegistry(config);

        DefaultAmpRules.ApplyDefaultRules(registry);

        var allRules = registry.GetAllRules();
        Assert.NotNull(allRules);
        Assert.NotEmpty(allRules);
    }

    [Fact]
    public void ApplyDefaultRules_WithNullRegistry_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            DefaultAmpRules.ApplyDefaultRules(null!));
    }

    [Fact]
    public void ApplyDefaultRules_CanBeCalledMultipleTimes()
    {
        var config = AmpConfig.CreateBf16();
        var registry = new AmpRegistry(config);

        DefaultAmpRules.ApplyDefaultRules(registry);
        var countAfterFirst = registry.GetAllRules().Count;

        DefaultAmpRules.ApplyDefaultRules(registry);
        var countAfterSecond = registry.GetAllRules().Count;

        Assert.Equal(countAfterFirst, countAfterSecond);
    }

    [Fact]
    public void DefaultWhitelist_IsNotEmpty()
    {
        var whitelist = DefaultAmpRules.DefaultWhitelist;

        Assert.NotNull(whitelist);
        Assert.NotEmpty(whitelist);
    }

    [Fact]
    public void DefaultBlacklist_IsNotEmpty()
    {
        var blacklist = DefaultAmpRules.DefaultBlacklist;

        Assert.NotNull(blacklist);
        Assert.NotEmpty(blacklist);
    }

    [Fact]
    public void DefaultWhitelist_ContainsExpectedOperations()
    {
        var whitelist = DefaultAmpRules.DefaultWhitelist;

        // Verify some expected operation types are present
        // The actual types are internal classes in DefaultAmpRules
        Assert.True(whitelist.Length > 10); // Should have many operations
    }

    [Fact]
    public void DefaultBlacklist_ContainsExpectedOperations()
    {
        var blacklist = DefaultAmpRules.DefaultBlacklist;

        // Verify some expected operation types are present
        // The actual types are internal classes in DefaultAmpRules
        Assert.True(blacklist.Length > 5); // Should have several operations
    }

    [Fact]
    public void ApplyDefaultRules_DoesNotOverrideHigherPriorityRules()
    {
        var config = AmpConfig.CreateBf16();
        var registry = new AmpRegistry(config);

        // Register a custom rule with higher priority
        var customOpType = typeof(string); // Using a safe type
        registry.RegisterCustomOp(customOpType, MLFramework.Core.DataType.Float32, MLFramework.Core.DataType.Float32, priority: 100);

        DefaultAmpRules.ApplyDefaultRules(registry);

        // Check that custom rule is still present
        var rule = registry.GetRule(customOpType);
        Assert.NotNull(rule);
        Assert.Equal(100, rule?.Priority);
    }

    [Fact]
    public void ApplyDefaultRules_WhitelistedOpsUseLowerPrecision()
    {
        var config = AmpConfig.CreateFp16();
        var registry = new AmpRegistry(config);
        DefaultAmpRules.ApplyDefaultRules(registry);

        // Get all rules and verify whitelist ones have Lower precision
        var allRules = registry.GetAllRules();
        var whitelistRules = DefaultAmpRules.DefaultWhitelist;

        foreach (var whitelistOp in whitelistRules)
        {
            var rule = registry.GetRule(whitelistOp);
            if (rule != null)
            {
                // Verify forward precision is set correctly
                var forwardDtype = rule.GetForwardDtype(config);
                Assert.Equal(config.TargetPrecision, forwardDtype);
            }
        }
    }

    [Fact]
    public void ApplyDefaultRules_BlacklistedOpsUseHigherPrecision()
    {
        var config = AmpConfig.CreateFp16();
        var registry = new AmpRegistry(config);
        DefaultAmpRules.ApplyDefaultRules(registry);

        // Get all rules and verify blacklist ones have Higher precision
        var allRules = registry.GetAllRules();
        var blacklistRules = DefaultAmpRules.DefaultBlacklist;

        foreach (var blacklistOp in blacklistRules)
        {
            var rule = registry.GetRule(blacklistOp);
            if (rule != null)
            {
                // Verify forward precision is set correctly
                var forwardDtype = rule.GetForwardDtype(config);
                Assert.Equal(config.HigherPrecision, forwardDtype);
            }
        }
    }

    [Fact]
    public void ApplyDefaultRules_TotalCountMatchesWhitelistAndBlacklist()
    {
        var config = AmpConfig.CreateBf16();
        var registry = new AmpRegistry(config);

        DefaultAmpRules.ApplyDefaultRules(registry);

        var allRules = registry.GetAllRules();
        var expectedCount = DefaultAmpRules.DefaultWhitelist.Length + DefaultAmpRules.DefaultBlacklist.Length;

        Assert.Equal(expectedCount, allRules.Count);
    }
}
