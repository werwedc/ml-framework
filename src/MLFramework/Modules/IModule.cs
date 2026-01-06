namespace MLFramework.Modules;

/// <summary>
/// Base interface for neural network modules.
/// </summary>
public interface IModule
{
    /// <summary>
    /// Gets the type of the module.
    /// </summary>
    string ModuleType { get; }
}
