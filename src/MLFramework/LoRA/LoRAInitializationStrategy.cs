namespace MLFramework.LoRA
{
    /// <summary>
    /// Defines the initialization strategy for LoRA adapter weights
    /// </summary>
    public enum LoRAInitializationStrategy
    {
        /// <summary>
        /// Initialize A with Kaiming normal, B with zeros (standard LoRA approach)
        /// </summary>
        Standard,

        /// <summary>
        /// Initialize both matrices with Xavier uniform
        /// </summary>
        Xavier,

        /// <summary>
        /// Initialize all weights to zero (start with zero perturbation)
        /// </summary>
        Zero
    }
}
