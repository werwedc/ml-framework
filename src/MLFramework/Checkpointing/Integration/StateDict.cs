namespace MachineLearning.Checkpointing;

using System.Collections;

/// <summary>
/// A dictionary that maps tensor names to their tensor values
/// </summary>
public class StateDict : IDictionary<string, ITensor>, IEnumerable<KeyValuePair<string, ITensor>>
{
    private readonly Dictionary<string, ITensor> _data;

    public StateDict()
    {
        _data = new Dictionary<string, ITensor>();
    }

    public ITensor this[string key]
    {
        get => _data[key];
        set => _data[key] = value;
    }

    public ICollection<string> Keys => _data.Keys;
    public ICollection<ITensor> Values => _data.Values;
    public int Count => _data.Count;
    public bool IsReadOnly => ((ICollection<KeyValuePair<string, ITensor>>)_data).IsReadOnly;

    public void Add(string key, ITensor value) => _data.Add(key, value);
    public void Add(KeyValuePair<string, ITensor> item) => ((ICollection<KeyValuePair<string, ITensor>>)_data).Add(item);
    public void Clear() => _data.Clear();
    public bool Contains(KeyValuePair<string, ITensor> item) => ((ICollection<KeyValuePair<string, ITensor>>)_data).Contains(item);
    public bool ContainsKey(string key) => _data.ContainsKey(key);
    public void CopyTo(KeyValuePair<string, ITensor>[] array, int arrayIndex) => ((ICollection<KeyValuePair<string, ITensor>>)_data).CopyTo(array, arrayIndex);
    public IEnumerator<KeyValuePair<string, ITensor>> GetEnumerator() => _data.GetEnumerator();
    public bool Remove(string key) => _data.Remove(key);
    public bool Remove(KeyValuePair<string, ITensor> item) => ((ICollection<KeyValuePair<string, ITensor>>)_data).Remove(item);
    public bool TryGetValue(string key, out ITensor value) => _data.TryGetValue(key, out value);
    IEnumerator IEnumerable.GetEnumerator() => _data.GetEnumerator();

    public void Deconstruct(out IReadOnlyDictionary<string, ITensor> items)
    {
        items = _data;
    }

    /// <summary>
    /// Get a tensor by key, throwing if not found
    /// </summary>
    public ITensor GetTensor(string key)
    {
        if (string.IsNullOrWhiteSpace(key))
            throw new ArgumentException("Key cannot be empty", nameof(key));

        if (!_data.TryGetValue(key, out var tensor))
            throw new KeyNotFoundException($"Tensor '{key}' not found in state dictionary");

        return tensor;
    }

    /// <summary>
    /// Get a tensor by key, returning null if not found
    /// </summary>
    public ITensor? GetTensorOrNull(string key)
    {
        if (string.IsNullOrWhiteSpace(key))
            throw new ArgumentException("Key cannot be empty", nameof(key));

        _data.TryGetValue(key, out var tensor);
        return tensor;
    }
}
