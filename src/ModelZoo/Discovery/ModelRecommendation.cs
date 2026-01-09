using System;
using System.Collections.Generic;

namespace MLFramework.ModelZoo.Discovery
{
    /// <summary>
    /// Represents a recommendation result from the model recommendation engine.
    /// </summary>
    public class ModelRecommendation
    {
        /// <summary>
        /// Gets or sets the recommended model.
        /// </summary>
        public ModelMetadata Model { get; set; } = null!;

        /// <summary>
        /// Gets or sets the explanation for this recommendation.
        /// </summary>
        public string Reason { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the estimated inference latency in milliseconds.
        /// </summary>
        public float EstimatedLatency { get; set; }

        /// <summary>
        /// Gets or sets the estimated memory usage in bytes.
        /// </summary>
        public float EstimatedMemory { get; set; }

        /// <summary>
        /// Gets or sets the compatibility score (0.0 to 1.0).
        /// Higher scores indicate better fit with constraints.
        /// </summary>
        public double CompatibilityScore { get; set; }

        /// <summary>
        /// Gets or sets alternative recommendations.
        /// </summary>
        public List<ModelMetadata> Alternatives { get; set; } = new List<ModelMetadata>();

        /// <summary>
        /// Initializes a new instance of the ModelRecommendation class.
        /// </summary>
        public ModelRecommendation()
        {
        }

        /// <summary>
        /// Initializes a new instance of the ModelRecommendation class.
        /// </summary>
        /// <param name="model">The recommended model.</param>
        /// <param name="reason">The explanation for recommendation.</param>
        /// <param name="estimatedLatency">Estimated inference latency in ms.</param>
        /// <param name="estimatedMemory">Estimated memory usage.</param>
        /// <param name="compatibilityScore">Compatibility score (0-1).</param>
        public ModelRecommendation(
            ModelMetadata model,
            string reason,
            float estimatedLatency,
            float estimatedMemory,
            double compatibilityScore)
        {
            Model = model ?? throw new ArgumentNullException(nameof(model));
            Reason = reason ?? throw new ArgumentNullException(nameof(reason));
            EstimatedLatency = estimatedLatency;
            EstimatedMemory = estimatedMemory;
            CompatibilityScore = compatibilityScore;
        }

        /// <summary>
        /// Checks if this recommendation meets all specified constraints.
        /// </summary>
        /// <param name="constraints">The constraints to check.</param>
        /// <returns>True if all constraints are satisfied.</returns>
        public bool SatisfiesConstraints(ModelConstraints constraints)
        {
            if (constraints == null)
                return true;

            // Check latency constraint
            if (constraints.MaxLatency.HasValue && EstimatedLatency > constraints.MaxLatency.Value)
                return false;

            // Check memory constraint
            if (constraints.MaxMemory.HasValue && EstimatedMemory > constraints.MaxMemory.Value)
                return false;

            // Check accuracy constraint
            if (constraints.MinAccuracy.HasValue)
            {
                if (Model.PerformanceMetrics.TryGetValue("accuracy", out var accuracy) &&
                    accuracy < constraints.MinAccuracy.Value)
                    return false;
            }

            // Check file size constraint
            if (constraints.MaxFileSize.HasValue && Model.FileSizeBytes > constraints.MaxFileSize.Value)
                return false;

            return true;
        }

        /// <summary>
        /// Gets a string summary of the recommendation.
        /// </summary>
        /// <returns>A summary string.</returns>
        public string GetSummary()
        {
            return $"Model: {Model.Name} v{Model.Version} | " +
                   $"Score: {CompatibilityScore:P2} | " +
                   $"Latency: {EstimatedLatency:F2}ms | " +
                   $"Memory: {(EstimatedMemory / 1024 / 1024):F2}MB";
        }
    }
}
