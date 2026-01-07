using RitterFramework.Core.Tensor;

namespace RitterFramework.Core.LoRA
{
    /// <summary>
    /// Base interface for all neural network modules.
    /// </summary>
    public interface IModule
    {
        /// <summary>
        /// Name of the module (for identification).
        /// </summary>
        string Name { get; set; }
    }
}
