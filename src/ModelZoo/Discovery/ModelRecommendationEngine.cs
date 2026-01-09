using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.ModelZoo.Discovery
{
    /// <summary>
    /// Provides intelligent model recommendations based on input characteristics and task constraints.
    /// </summary>
    public class ModelRecommendationEngine
    {
        private readonly ModelRegistry _registry;
        private readonly LatencyEstimator _latencyEstimator;
        private readonly MemoryEstimator _memoryEstimator;

        /// <summary>
        /// Initializes a new instance of the ModelRecommendationEngine class.
        /// </summary>
        /// <param name="registry">The model registry to query.</param>
        public ModelRecommendationEngine(ModelRegistry registry)
        {
            _registry = registry ?? throw new ArgumentNullException(nameof(registry));
            _latencyEstimator = new LatencyEstimator();
            _memoryEstimator = new MemoryEstimator();
        }

        /// <summary>
        /// Gets a single best recommendation for the given input shape and task.
        /// </summary>
        /// <param name="inputShape">The input tensor shape.</param>
        /// <param name="task">The target task type.</param>
        /// <param name="constraints">Additional model constraints.</param>
        /// <returns>The best model recommendation.</returns>
        public ModelRecommendation RecommendFor(
            Shape inputShape,
            TaskType task,
            ModelConstraints? constraints = null)
        {
            // Build constraints if not provided
            var fullConstraints = constraints ?? new ModelConstraints(inputShape, task);

            // Get candidate models
            var candidates = GetCandidateModels(fullConstraints);

            if (candidates.Count == 0)
                throw new InvalidOperationException($"No models found matching task: {task}");

            // Score and rank models
            var scoredModels = ScoreModels(candidates, fullConstraints, RecommendationStrategy.Balanced);

            if (scoredModels.Count == 0)
                throw new InvalidOperationException($"No models satisfy the given constraints");

            // Get top recommendation
            var topModel = scoredModels[0];
            var alternatives = scoredModels.Skip(1).Take(5).Select(s => s.Model).ToList();

            return new ModelRecommendation(
                topModel.Model,
                GenerateReason(topModel, fullConstraints),
                topModel.EstimatedLatency,
                topModel.EstimatedMemory,
                topModel.CompatibilityScore)
            {
                Alternatives = alternatives
            };
        }

        /// <summary>
        /// Gets top N recommendations for the given input shape and task.
        /// </summary>
        /// <param name="inputShape">The input tensor shape.</param>
        /// <param name="task">The target task type.</param>
        /// <param name="topN">Number of recommendations to return.</param>
        /// <param name="constraints">Additional model constraints.</param>
        /// <returns>List of model recommendations.</returns>
        public List<ModelRecommendation> RecommendMultiple(
            Shape inputShape,
            TaskType task,
            int topN,
            ModelConstraints? constraints = null)
        {
            if (topN <= 0)
                throw new ArgumentException("topN must be positive", nameof(topN));

            // Build constraints if not provided
            var fullConstraints = constraints ?? new ModelConstraints(inputShape, task);

            // Get candidate models
            var candidates = GetCandidateModels(fullConstraints);

            if (candidates.Count == 0)
                throw new InvalidOperationException($"No models found matching task: {task}");

            // Score and rank models
            var scoredModels = ScoreModels(candidates, fullConstraints, RecommendationStrategy.Balanced);

            // Return top N recommendations
            return scoredModels.Take(topN).Select(s =>
                new ModelRecommendation(
                    s.Model,
                    GenerateReason(s, fullConstraints),
                    s.EstimatedLatency,
                    s.EstimatedMemory,
                    s.CompatibilityScore)
                {
                    Alternatives = scoredModels
                        .Where((m, i) => i > 0 && i <= 5)
                        .Select(m => m.Model)
                        .ToList()
                }).ToList();
        }

        /// <summary>
        /// Gets recommendations based on constraints only.
        /// </summary>
        /// <param name="constraints">The model constraints.</param>
        /// <returns>List of model recommendations.</returns>
        public List<ModelRecommendation> RecommendForConstraints(ModelConstraints constraints)
        {
            if (constraints == null)
                throw new ArgumentNullException(nameof(constraints));

            // Get candidate models
            var candidates = GetCandidateModels(constraints);

            if (candidates.Count == 0)
                throw new InvalidOperationException($"No models found matching the given constraints");

            // Score and rank models
            var scoredModels = ScoreModels(candidates, constraints, RecommendationStrategy.Balanced);

            // Return all matching models as recommendations
            return scoredModels.Select(s =>
                new ModelRecommendation(
                    s.Model,
                    GenerateReason(s, constraints),
                    s.EstimatedLatency,
                    s.EstimatedMemory,
                    s.CompatibilityScore)
                {
                    Alternatives = scoredModels
                        .Where((m, i) => i > 0 && i <= 5)
                        .Select(m => m.Model)
                        .ToList()
                }).ToList();
        }

        /// <summary>
        /// Gets alternatives to a specific model.
        /// </summary>
        /// <param name="modelName">The name of the model.</param>
        /// <param name="count">Number of alternatives to return.</param>
        /// <returns>List of alternative models.</returns>
        public List<ModelMetadata> GetAlternatives(string modelName, int count)
        {
            if (string.IsNullOrWhiteSpace(modelName))
                throw new ArgumentNullException(nameof(modelName));

            if (count <= 0)
                throw new ArgumentException("count must be positive", nameof(count));

            // Get the reference model
            var referenceModel = _registry.GetLatestVersion(modelName);
            if (referenceModel == null)
                throw new InvalidOperationException($"Model not found: {modelName}");

            // Find similar models (same task)
            var similarModels = _registry.ListByTask(referenceModel.Task)
                .Where(m => m.Name != referenceModel.Name)
                .ToList();

            // Score based on similarity to reference model
            var scoredModels = similarModels
                .Select(m => new
                {
                    Model = m,
                    SimilarityScore = CalculateSimilarityScore(referenceModel, m)
                })
                .OrderByDescending(s => s.SimilarityScore)
                .Take(count)
                .Select(s => s.Model)
                .ToList();

            return scoredModels;
        }

        /// <summary>
        /// Gets recommendations using a specific strategy.
        /// </summary>
        /// <param name="inputShape">The input tensor shape.</param>
        /// <param name="task">The target task type.</param>
        /// <param name="strategy">The recommendation strategy.</param>
        /// <param name="topN">Number of recommendations to return.</param>
        /// <param name="constraints">Additional model constraints.</param>
        /// <returns>List of model recommendations.</returns>
        public List<ModelRecommendation> RecommendWithStrategy(
            Shape inputShape,
            TaskType task,
            RecommendationStrategy strategy,
            int topN = 1,
            ModelConstraints? constraints = null)
        {
            if (topN <= 0)
                throw new ArgumentException("topN must be positive", nameof(topN));

            // Build constraints if not provided
            var fullConstraints = constraints ?? new ModelConstraints(inputShape, task);

            // Get candidate models
            var candidates = GetCandidateModels(fullConstraints);

            if (candidates.Count == 0)
                throw new InvalidOperationException($"No models found matching task: {task}");

            // Score and rank models with specified strategy
            var scoredModels = ScoreModels(candidates, fullConstraints, strategy);

            // Return top N recommendations
            return scoredModels.Take(topN).Select(s =>
                new ModelRecommendation(
                    s.Model,
                    GenerateReason(s, fullConstraints, strategy),
                    s.EstimatedLatency,
                    s.EstimatedMemory,
                    s.CompatibilityScore)
                {
                    Alternatives = scoredModels
                        .Where((m, i) => i > 0 && i <= 5)
                        .Select(m => m.Model)
                        .ToList()
                }).ToList();
        }

        /// <summary>
        /// Gets candidate models matching the given constraints.
        /// </summary>
        /// <param name="constraints">The model constraints.</param>
        /// <returns>List of candidate models.</returns>
        private List<ModelMetadata> GetCandidateModels(ModelConstraints constraints)
        {
            // Start with models matching the task
            var candidates = _registry.ListByTask(constraints.Task).ToList();

            // Filter by input shape compatibility if specified
            if (constraints.InputShape != null)
            {
                var inputShape = new Shape(constraints.InputShape.InputShape);
                candidates = candidates.Where(m =>
                {
                    var modelInputShape = new Shape(m.InputShape);
                    return modelInputShape.IsCompatibleWith(inputShape);
                }).ToList();
            }

            // Filter by file size if specified
            if (constraints.MaxFileSize.HasValue)
            {
                candidates = candidates.Where(m => m.FileSizeBytes <= constraints.MaxFileSize.Value).ToList();
            }

            return candidates;
        }

        /// <summary>
        /// Scores models based on constraints and recommendation strategy.
        /// </summary>
        /// <param name="models">The candidate models.</param>
        /// <param name="constraints">The model constraints.</param>
        /// <param name="strategy">The recommendation strategy.</param>
        /// <returns>List of scored models sorted by score.</returns>
        private List<ScoredModel> ScoreModels(
            List<ModelMetadata> models,
            ModelConstraints constraints,
            RecommendationStrategy strategy)
        {
            var scoredModels = new List<ScoredModel>();

            foreach (var model in models)
            {
                // Determine device
                var device = constraints.Device ?? DeviceType.CPU;
                var batchSize = constraints.BatchSize ?? 1;
                var inputShape = constraints.InputShape;

                // Estimate latency and memory
                float latency = inputShape != null
                    ? _latencyEstimator.EstimateLatencyWithShape(model, device, inputShape, batchSize)
                    : _latencyEstimator.EstimateLatency(model, device, batchSize);

                float memory = inputShape != null
                    ? _memoryEstimator.EstimateMemoryWithShape(model, device, inputShape, batchSize)
                    : _memoryEstimator.EstimateMemory(model, device, batchSize);

                // Calculate compatibility score
                double compatibilityScore = CalculateCompatibilityScore(
                    model, constraints, latency, memory, strategy);

                scoredModels.Add(new ScoredModel
                {
                    Model = model,
                    EstimatedLatency = latency,
                    EstimatedMemory = memory,
                    CompatibilityScore = compatibilityScore
                });
            }

            // Sort by score (descending)
            return scoredModels.OrderByDescending(s => s.CompatibilityScore).ToList();
        }

        /// <summary>
        /// Calculates the compatibility score for a model.
        /// </summary>
        /// <param name="model">The model metadata.</param>
        /// <param name="constraints">The model constraints.</param>
        /// <param name="estimatedLatency">Estimated latency.</param>
        /// <param name="estimatedMemory">Estimated memory.</param>
        /// <param name="strategy">The recommendation strategy.</param>
        /// <returns>Compatibility score (0.0 to 1.0).</returns>
        private double CalculateCompatibilityScore(
            ModelMetadata model,
            ModelConstraints constraints,
            float estimatedLatency,
            float estimatedMemory,
            RecommendationStrategy strategy)
        {
            double score = 0.0;
            double totalWeight = 0.0;

            // Input shape compatibility
            if (constraints.InputShape != null)
            {
                var inputShape = new Shape(constraints.InputShape.InputShape);
                var modelInputShape = new Shape(model.InputShape);

                double shapeScore = modelInputShape.MatchesExactly(inputShape) ? 1.0 :
                                    modelInputShape.IsCompatibleWith(inputShape) ? 0.8 : 0.0;

                score += shapeScore * 0.3; // 30% weight
                totalWeight += 0.3;
            }

            // Latency constraint
            if (constraints.MaxLatency.HasValue)
            {
                double latencyScore = Math.Max(0, (constraints.MaxLatency.Value - estimatedLatency) / constraints.MaxLatency.Value);

                // Adjust based on strategy
                if (strategy == RecommendationStrategy.PerformanceFirst)
                {
                    score += latencyScore * 0.4; // Higher weight for performance
                    totalWeight += 0.4;
                }
                else
                {
                    score += latencyScore * 0.2;
                    totalWeight += 0.2;
                }
            }
            else
            {
                // Reward lower latency even without constraint
                double latencyScore = 1.0 / (1.0 + estimatedLatency / 10.0); // Normalize
                score += latencyScore * 0.2;
                totalWeight += 0.2;
            }

            // Memory constraint
            if (constraints.MaxMemory.HasValue)
            {
                double memoryScore = Math.Max(0, (constraints.MaxMemory.Value - estimatedMemory) / constraints.MaxMemory.Value);

                // Adjust based on strategy
                if (strategy == RecommendationStrategy.MemoryConstrained || strategy == RecommendationStrategy.EdgeDeployment)
                {
                    score += memoryScore * 0.4; // Higher weight for memory
                    totalWeight += 0.4;
                }
                else
                {
                    score += memoryScore * 0.15;
                    totalWeight += 0.15;
                }
            }
            else
            {
                // Reward smaller models even without constraint
                double memoryScore = 1.0 / (1.0 + estimatedMemory / (100 * 1024 * 1024)); // Normalize against 100MB
                score += memoryScore * 0.15;
                totalWeight += 0.15;
            }

            // Accuracy constraint
            if (constraints.MinAccuracy.HasValue && model.PerformanceMetrics.TryGetValue("accuracy", out var accuracy))
            {
                double accuracyScore = Math.Min(1.0, accuracy / constraints.MinAccuracy.Value);

                // Adjust based on strategy
                if (strategy == RecommendationStrategy.AccuracyFirst)
                {
                    score += accuracyScore * 0.4; // Higher weight for accuracy
                    totalWeight += 0.4;
                }
                else
                {
                    score += accuracyScore * 0.2;
                    totalWeight += 0.2;
                }
            }
            else if (model.PerformanceMetrics.TryGetValue("accuracy", out var modelAccuracy))
            {
                // Reward higher accuracy even without constraint
                score += modelAccuracy * 0.2;
                totalWeight += 0.2;
            }

            // File size constraint
            if (constraints.MaxFileSize.HasValue)
            {
                double fileSizeScore = Math.Max(0, (constraints.MaxFileSize.Value - model.FileSizeBytes) / (double)constraints.MaxFileSize.Value);
                score += fileSizeScore * 0.1;
                totalWeight += 0.1;
            }

            // Normalize score
            if (totalWeight > 0)
            {
                score /= totalWeight;
            }

            return Math.Min(1.0, Math.Max(0.0, score));
        }

        /// <summary>
        /// Calculates similarity score between two models.
        /// </summary>
        /// <param name="model1">First model.</param>
        /// <param name="model2">Second model.</param>
        /// <returns>Similarity score (0.0 to 1.0).</returns>
        private double CalculateSimilarityScore(ModelMetadata model1, ModelMetadata model2)
        {
            double score = 0.0;

            // Same architecture - higher score
            if (model1.Architecture.Equals(model2.Architecture, StringComparison.OrdinalIgnoreCase))
                score += 0.3;

            // Similar parameter count
            long paramDiff = Math.Abs(model1.NumParameters - model2.NumParameters);
            double paramScore = 1.0 - (paramDiff / (double)Math.Max(model1.NumParameters, model2.NumParameters));
            score += paramScore * 0.4;

            // Similar accuracy
            if (model1.PerformanceMetrics.TryGetValue("accuracy", out var acc1) &&
                model2.PerformanceMetrics.TryGetValue("accuracy", out var acc2))
            {
                double accDiff = Math.Abs(acc1 - acc2);
                double accScore = 1.0 - accDiff;
                score += accScore * 0.3;
            }

            return Math.Min(1.0, Math.Max(0.0, score));
        }

        /// <summary>
        /// Generates a human-readable reason for the recommendation.
        /// </summary>
        /// <param name="scoredModel">The scored model.</param>
        /// <param name="constraints">The model constraints.</param>
        /// <param name="strategy">The recommendation strategy.</param>
        /// <returns>A reason string.</returns>
        private string GenerateReason(ScoredModel scoredModel, ModelConstraints constraints, RecommendationStrategy? strategy = null)
        {
            var reasons = new List<string>();

            // Task match
            reasons.Add($"Matches task: {constraints.Task}");

            // Performance characteristics
            if (scoredModel.EstimatedLatency < 10)
                reasons.Add("Low latency");
            else if (scoredModel.EstimatedLatency < 50)
                reasons.Add("Moderate latency");
            else
                reasons.Add("Higher latency");

            if (scoredModel.EstimatedMemory < 50 * 1024 * 1024)
                reasons.Add("Low memory footprint");
            else if (scoredModel.EstimatedMemory < 200 * 1024 * 1024)
                reasons.Add("Moderate memory usage");
            else
                reasons.Add("Higher memory usage");

            // Accuracy
            if (scoredModel.Model.PerformanceMetrics.TryGetValue("accuracy", out var accuracy))
                reasons.Add($"Accuracy: {accuracy:P2}");

            // Strategy-specific reasons
            if (strategy.HasValue)
            {
                switch (strategy.Value)
                {
                    case RecommendationStrategy.AccuracyFirst:
                        reasons.Add("Prioritized for accuracy");
                        break;
                    case RecommendationStrategy.PerformanceFirst:
                        reasons.Add("Prioritized for performance");
                        break;
                    case RecommendationStrategy.MemoryConstrained:
                        reasons.Add("Prioritized for memory efficiency");
                        break;
                    case RecommendationStrategy.EdgeDeployment:
                        reasons.Add("Optimized for edge deployment");
                        break;
                    case RecommendationStrategy.Balanced:
                        reasons.Add("Balanced trade-off");
                        break;
                }
            }

            return string.Join(", ", reasons);
        }

        /// <summary>
        /// Internal class to track scored models.
        /// </summary>
        private class ScoredModel
        {
            public ModelMetadata Model { get; set; } = null!;
            public float EstimatedLatency { get; set; }
            public float EstimatedMemory { get; set; }
            public double CompatibilityScore { get; set; }
        }
    }
}
