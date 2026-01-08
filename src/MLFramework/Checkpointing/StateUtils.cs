namespace MachineLearning.Checkpointing;

/// <summary>
/// Utility methods for working with StateDict
/// </summary>
public static class StateUtils
{
    /// <summary>
    /// Check if two state dicts have matching keys
    /// </summary>
    public static bool KeysMatch(StateDict state1, StateDict state2)
    {
        if (state1 == null || state2 == null)
            return false;

        if (state1.Count != state2.Count)
            return false;

        var keys1 = new HashSet<string>(state1.Keys);
        var keys2 = new HashSet<string>(state2.Keys);
        return keys1.SetEquals(keys2);
    }

    /// <summary>
    /// Get missing keys from state2 compared to state1
    /// </summary>
    public static HashSet<string> GetMissingKeys(StateDict state1, StateDict state2)
    {
        var missing = new HashSet<string>();

        if (state1 == null)
            return missing;

        var keys2 = state2 != null ? new HashSet<string>(state2.Keys) : new HashSet<string>();

        foreach (var key in state1.Keys)
        {
            if (!keys2.Contains(key))
                missing.Add(key);
        }

        return missing;
    }

    /// <summary>
    /// Get extra keys in state2 compared to state1
    /// </summary>
    public static HashSet<string> GetUnexpectedKeys(StateDict state1, StateDict state2)
    {
        var unexpected = new HashSet<string>();

        if (state2 == null)
            return unexpected;

        var keys1 = state1 != null ? new HashSet<string>(state1.Keys) : new HashSet<string>();

        foreach (var key in state2.Keys)
        {
            if (!keys1.Contains(key))
                unexpected.Add(key);
        }

        return unexpected;
    }

    /// <summary>
    /// Verify tensor shapes match between two state dicts
    /// </summary>
    public static bool ShapesMatch(StateDict state1, StateDict state2)
    {
        if (state1 == null || state2 == null)
            return false;

        if (!KeysMatch(state1, state2))
            return false;

        foreach (var (key, tensor1) in state1)
        {
            if (!state2.TryGetValue(key, out var tensor2))
                return false;

            if (!tensor1.Shape.SequenceEqual(tensor2.Shape))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Check if two tensor shapes match
    /// </summary>
    public static bool ShapesMatch(long[] shape1, long[] shape2)
    {
        if (shape1 == null || shape2 == null)
            return shape1 == shape2;

        if (shape1.Length != shape2.Length)
            return false;

        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Verify data types match between two state dicts
    /// </summary>
    public static bool DataTypesMatch(StateDict state1, StateDict state2)
    {
        if (state1 == null || state2 == null)
            return false;

        if (!KeysMatch(state1, state2))
            return false;

        foreach (var (key, tensor1) in state1)
        {
            if (!state2.TryGetValue(key, out var tensor2))
                return false;

            if (tensor1.DataType != tensor2.DataType)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Get the total size (in elements) of all tensors in a state dict
    /// </summary>
    public static long GetTotalSize(StateDict state)
    {
        if (state == null)
            return 0;

        long totalSize = 0;
        foreach (var (_, tensor) in state)
        {
            if (tensor.Shape.Length > 0)
            {
                long tensorSize = 1;
                foreach (var dim in tensor.Shape)
                    tensorSize *= dim;
                totalSize += tensorSize;
            }
        }
        return totalSize;
    }

    /// <summary>
    /// Get the total size (in bytes) of all tensors in a state dict
    /// </summary>
    public static long GetTotalSizeInBytes(StateDict state)
    {
        if (state == null)
            return 0;

        long totalBytes = 0;
        foreach (var (_, tensor) in state)
        {
            totalBytes += tensor.GetSizeInBytes();
        }
        return totalBytes;
    }

    /// <summary>
    /// Clone a state dict (shallow copy - tensors are not cloned)
    /// </summary>
    public static StateDict Clone(StateDict state)
    {
        if (state == null)
            return new StateDict();

        var cloned = new StateDict();
        foreach (var (key, value) in state)
        {
            cloned[key] = value;
        }
        return cloned;
    }

    /// <summary>
    /// Get a summary of the state dict contents
    /// </summary>
    public static string GetSummary(StateDict state)
    {
        if (state == null)
            return "StateDict: null";

        var summary = new System.Text.StringBuilder();
        summary.AppendLine($"StateDict: {state.Count} tensors");
        summary.AppendLine($"Total elements: {GetTotalSize(state)}");
        summary.AppendLine($"Total bytes: {GetTotalSizeInBytes(state)}");
        summary.AppendLine();
        summary.AppendLine("Tensors:");

        foreach (var (key, tensor) in state.OrderBy(k => k.Key))
        {
            summary.AppendLine($"  {key}: shape=[{string.Join(", ", tensor.Shape)}], dtype={tensor.DataType}");
        }

        return summary.ToString();
    }

    /// <summary>
    /// Filter a state dict to only include tensors matching a predicate
    /// </summary>
    public static StateDict Filter(StateDict state, Func<string, ITensor, bool> predicate)
    {
        if (state == null)
            return new StateDict();

        var filtered = new StateDict();
        foreach (var (key, tensor) in state)
        {
            if (predicate(key, tensor))
            {
                filtered[key] = tensor;
            }
        }
        return filtered;
    }

    /// <summary>
    /// Filter a state dict by key prefix
    /// </summary>
    public static StateDict FilterByPrefix(StateDict state, string prefix)
    {
        if (state == null)
            return new StateDict();

        return Filter(state, (key, _) => key.StartsWith(prefix));
    }

    /// <summary>
    /// Remove a prefix from all keys in a state dict
    /// </summary>
    public static StateDict RemovePrefix(StateDict state, string prefix)
    {
        if (state == null)
            return new StateDict();

        var result = new StateDict();
        foreach (var (key, tensor) in state)
        {
            if (key.StartsWith(prefix))
            {
                result[key.Substring(prefix.Length)] = tensor;
            }
            else
            {
                result[key] = tensor;
            }
        }
        return result;
    }

    /// <summary>
    /// Add a prefix to all keys in a state dict
    /// </summary>
    public static StateDict AddPrefix(StateDict state, string prefix)
    {
        if (state == null)
            return new StateDict();

        var result = new StateDict();
        foreach (var (key, tensor) in state)
        {
            result[$"{prefix}{key}"] = tensor;
        }
        return result;
    }
}
