namespace MLFramework.Schedulers;

/// <summary>
/// Represents a dictionary of scheduler state for checkpointing.
/// Uses key-value pairs for serializable state.
/// </summary>
public class StateDict
{
    private Dictionary<string, object> _state;

    public StateDict()
    {
        _state = new Dictionary<string, object>();
    }

    public StateDict(Dictionary<string, object> state)
    {
        _state = state ?? new Dictionary<string, object>();
    }

    public T Get<T>(string key, T defaultValue = default)
    {
        if (_state.TryGetValue(key, out var value))
        {
            return (T)value;
        }
        return defaultValue;
    }

    public void Set<T>(string key, T value)
    {
        _state[key] = value;
    }

    public bool ContainsKey(string key)
    {
        return _state.ContainsKey(key);
    }

    public Dictionary<string, object> ToDictionary()
    {
        return new Dictionary<string, object>(_state);
    }
}
