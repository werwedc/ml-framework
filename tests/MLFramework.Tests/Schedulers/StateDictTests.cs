using MLFramework.Schedulers;
using Xunit;

namespace MLFramework.Tests.Schedulers;

public class StateDictTests
{
    [Fact]
    public void Constructor_Empty_CreatesEmptyDict()
    {
        var state = new StateDict();

        Assert.Empty(state.ToDictionary());
    }

    [Fact]
    public void Constructor_WithDict_CopiesDict()
    {
        var dict = new Dictionary<string, object>
        {
            { "step", 100 },
            { "lr", 0.01f }
        };

        var state = new StateDict(dict);

        Assert.Equal(100, state.Get<int>("step"));
        Assert.Equal(0.01f, state.Get<float>("lr"));
    }

    [Fact]
    public void Constructor_NullDict_CreatesEmptyDict()
    {
        var state = new StateDict(null);

        Assert.Empty(state.ToDictionary());
    }

    [Fact]
    public void Get_ReturnsStoredValue()
    {
        var state = new StateDict();
        state.Set("key1", 42);

        Assert.Equal(42, state.Get<int>("key1"));
    }

    [Fact]
    public void Get_WithDefaultValue_ReturnsDefaultWhenKeyNotFound()
    {
        var state = new StateDict();

        Assert.Equal(100, state.Get<int>("nonexistent", 100));
    }

    [Fact]
    public void Get_WithDefaultValue_ReturnsDefaultForNull()
    {
        var state = new StateDict();

        Assert.Equal(default(string), state.Get<string>("nonexistent"));
    }

    [Fact]
    public void Set_UpdatesValue()
    {
        var state = new StateDict();
        state.Set("key", 1);
        state.Set("key", 2);

        Assert.Equal(2, state.Get<int>("key"));
    }

    [Fact]
    public void ContainsKey_ReturnsTrueForExistingKey()
    {
        var state = new StateDict();
        state.Set("existing", 123);

        Assert.True(state.ContainsKey("existing"));
    }

    [Fact]
    public void ContainsKey_ReturnsFalseForNonExistentKey()
    {
        var state = new StateDict();

        Assert.False(state.ContainsKey("nonexistent"));
    }

    [Fact]
    public void ToDictionary_ReturnsCopy()
    {
        var state = new StateDict();
        state.Set("key1", 1);

        var dict1 = state.ToDictionary();
        state.Set("key2", 2);
        var dict2 = state.ToDictionary();

        Assert.Single(dict1);
        Assert.Equal(2, dict2.Count);
    }

    [Fact]
    public void Get_SupportsMultipleTypes()
    {
        var state = new StateDict();
        state.Set("int_val", 42);
        state.Set("float_val", 3.14f);
        state.Set("string_val", "test");

        Assert.Equal(42, state.Get<int>("int_val"));
        Assert.Equal(3.14f, state.Get<float>("float_val"));
        Assert.Equal("test", state.Get<string>("string_val"));
    }
}
