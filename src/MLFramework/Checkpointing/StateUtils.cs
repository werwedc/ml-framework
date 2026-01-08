namespace MachineLearning.Checkpointing;

/// <summary>
/// Utility methods for working with StateDict
/// </summary>
public static class StateUtils
{
    /// <summary>
    /// Check if two StateDicts have the same keys
    /// </summary>
    public static bool KeysMatch(StateDict state1, StateDict state2)
    {
        if (state1 == null || state2 == null)
            return false;

        if (state1.Count != state2.Count)
            return false;

        foreach (var key in state1.Keys)
        {
            if (!state2.ContainsKey(key))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Check if two StateDicts have matching tensor shapes
    /// </summary>
    public static bool ShapesMatch(StateDict state1, StateDict state2)
    {
        if (state1 == null || state2 == null)
            return false;

        if (!KeysMatch(state1, state2))
            return false;

        foreach (var key in state1.Keys)
        {
            var tensor1 = state1[key];
            var tensor2 = state2[key];

            if (!ShapesMatch(tensor1.Shape, tensor2.Shape))
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
}
