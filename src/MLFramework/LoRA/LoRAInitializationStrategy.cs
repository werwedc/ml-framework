namespace MLFramework.LoRA;

/// <summary>
/// Strategy for initializing LoRA adapter matrices.
/// </summary>
public enum LoRAInitializationStrategy
{
    /// <summary>
    /// Standard initialization using normal distribution.
    /// </summary>
    Standard,

    /// <summary>
    /// Kaiming/He initialization.
    /// </summary>
    Kaiming,

    /// <summary>
    /// Xavier/Glorot initialization.
    /// </summary>
    Xavier
}
