using System;

namespace MLFramework.ModelZoo.TransferLearning
{
    /// <summary>
    /// Defines weight initialization strategies for neural network layers.
    /// </summary>
    public enum WeightInitializationStrategy
    {
        /// <summary>
        /// Xavier (Glorot) initialization - optimal for sigmoid/tanh activations.
        /// Draws from a uniform distribution with bounds derived from input/output dimensions.
        /// </summary>
        Xavier,

        /// <summary>
        /// Kaiming (He) initialization - optimal for ReLU and its variants.
        /// Draws from a normal distribution scaled by the number of input units.
        /// </summary>
        Kaiming,

        /// <summary>
        /// Uniform initialization - draws from a uniform distribution in [-limit, limit].
        /// </summary>
        Uniform,

        /// <summary>
        /// Normal initialization - draws from a normal distribution with given mean and std.
        /// </summary>
        Normal,

        /// <summary>
        /// Truncated normal initialization - draws from a normal distribution but rejects values
        /// more than 2 standard deviations from the mean.
        /// </summary>
        TruncatedNormal,

        /// <summary>
        /// Orthogonal initialization - generates orthogonal weight matrices to preserve
        /// gradient flow in deep networks.
        /// </summary>
        Orthogonal
    }

    /// <summary>
    /// Extension methods for WeightInitializationStrategy.
    /// </summary>
    public static class WeightInitializationStrategyExtensions
    {
        /// <summary>
        /// Gets the default parameters for this initialization strategy.
        /// </summary>
        /// <returns>A tuple containing the default gain and any additional parameters.</returns>
        public static (float gain, float std) GetDefaultParameters(this WeightInitializationStrategy strategy)
        {
            return strategy switch
            {
                WeightInitializationStrategy.Xavier => (1.0f, 1.0f),
                WeightInitializationStrategy.Kaiming => (MathF.Sqrt(2.0f), 1.0f),
                WeightInitializationStrategy.Uniform => (1.0f, 0.02f),
                WeightInitializationStrategy.Normal => (1.0f, 0.02f),
                WeightInitializationStrategy.TruncatedNormal => (1.0f, 0.02f),
                WeightInitializationStrategy.Orthogonal => (1.0f, 1.0f),
                _ => (1.0f, 0.02f)
            };
        }

        /// <summary>
        /// Checks if this strategy is suitable for a given activation function.
        /// </summary>
        /// <param name="activation">The activation function type.</param>
        /// <returns>True if the strategy is recommended, false otherwise.</returns>
        public static bool IsRecommendedForActivation(this WeightInitializationStrategy strategy, string activation)
        {
            return strategy switch
            {
                WeightInitializationStrategy.Xavier => 
                    activation == "sigmoid" || activation == "tanh" || activation == "tanh_shrink",
                WeightInitializationStrategy.Kaiming => 
                    activation.Contains("relu") || activation == "leaky_relu" || activation == "elu",
                _ => true // Other strategies are generic
            };
        }
    }
}
