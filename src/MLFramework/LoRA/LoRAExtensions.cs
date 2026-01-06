using MLFramework.Modules;

namespace MLFramework.LoRA
{
    /// <summary>
    /// Extension methods for easily creating LoRA adapters
    /// </summary>
    public static class LoRAExtensions
    {
        /// <summary>
        /// Wraps a Linear layer with LoRA adapter
        /// </summary>
        public static LoRALinear AsLoRA(this Linear linear, int rank, float alpha,
                                         LoRAInitializationStrategy initialization = LoRAInitializationStrategy.Standard,
                                         float dropout = 0.0f, bool useBias = false)
        {
            return new LoRALinear(linear, rank, alpha, initialization, dropout, useBias);
        }
    }
}
