namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for objects that can save and load their state
/// </summary>
public interface IStateful
{
    /// <summary>
    /// Get the current state of the object
    /// </summary>
    StateDict GetStateDict();

    /// <summary>
    /// Load state into the object
    /// </summary>
    void LoadStateDict(StateDict state);
}
