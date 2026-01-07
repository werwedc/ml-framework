using System;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Metrics from pipeline validation
    /// </summary>
    public class ValidationMetrics
    {
        /// <summary>
        /// Gradient L2 norm difference (pipeline vs single-device)
        /// </summary>
        public float GradientDifference { get; }

        /// <summary>
        /// Parameter L2 norm difference (after sync)
        /// </summary>
        public float ParameterDifference { get; }

        /// <summary>
        /// Maximum activation value
        /// </summary>
        public float MaxActivation { get; }

        /// <summary>
        /// Minimum activation value
        /// </summary>
        public float MinActivation { get; }

        /// <summary>
        /// Maximum gradient value
        /// </summary>
        public float MaxGradient { get; }

        /// <summary>
        /// Minimum gradient value
        /// </summary>
        public float MinGradient { get; }

        /// <summary>
        /// Number of NaN/Inf values in activations
        /// </summary>
        public int NaNInfCount { get; }

        /// <summary>
        /// Memory usage per stage (in bytes)
        /// </summary>
        public long[] MemoryUsage { get; }

        public ValidationMetrics(
            float gradientDifference,
            float parameterDifference,
            float maxActivation,
            float minActivation,
            float maxGradient,
            float minGradient,
            int nanInfCount,
            long[] memoryUsage)
        {
            GradientDifference = gradientDifference;
            ParameterDifference = parameterDifference;
            MaxActivation = maxActivation;
            MinActivation = minActivation;
            MaxGradient = maxGradient;
            MinGradient = minGradient;
            NaNInfCount = nanInfCount;
            MemoryUsage = memoryUsage ?? throw new ArgumentNullException(nameof(memoryUsage));
        }

        /// <summary>
        /// Gets a string summary of validation metrics
        /// </summary>
        public override string ToString()
        {
            return $"ValidationMetrics:\n" +
                   $"  Gradient Diff: {GradientDifference:E4}\n" +
                   $"  Parameter Diff: {ParameterDifference:E4}\n" +
                   $"  Activation Range: [{MinActivation:F4}, {MaxActivation:F4}]\n" +
                   $"  Gradient Range: [{MinGradient:F4}, {MaxGradient:F4}]\n" +
                   $"  NaN/Inf Count: {NaNInfCount}\n" +
                   $"  Memory Usage per Stage: [{string.Join(", ", MemoryUsage)} bytes]";
        }
    }
}
