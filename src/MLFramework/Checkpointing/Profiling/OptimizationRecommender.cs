using System.Text.Json;

namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Generates optimization recommendations based on profiling data
/// </summary>
public class OptimizationRecommender
{
    /// <summary>
    /// Generates recommendations from profiling summary
    /// </summary>
    /// <param name="summary">Profiling summary</param>
    /// <returns>List of recommendations</returns>
    public List<OptimizationRecommendation> GenerateRecommendations(ProfilingSummary summary)
    {
        var recommendations = new List<OptimizationRecommendation>();

        // Check for layers with high recomputation cost
        var highRecomputeLayers = summary.LayerProfiles
            .Where(p => p.RecomputeCount > p.CheckpointCount * 2)
            .OrderByDescending(p => p.TotalRecomputeTimeMs)
            .Take(3)
            .ToList();

        foreach (var layer in highRecomputeLayers)
        {
            recommendations.Add(new OptimizationRecommendation
            {
                Type = RecommendationType.ReduceRecomputation,
                Priority = RecommendationPriority.High,
                Title = $"Reduce recomputation for layer '{layer.LayerId}'",
                Description = $"Layer '{layer.LayerId}' is recomputed {layer.RecomputeCount} times " +
                              $"but only checkpointed {layer.CheckpointCount} times. " +
                              $"Consider increasing checkpoint frequency or using selective checkpointing.",
                AffectedLayerId = layer.LayerId,
                ExpectedImpact = $"Could save {FormatMs(layer.TotalRecomputeTimeMs)} of recomputation time"
            });
        }

        // Check for layers with low cache hit rate
        var lowCacheHitLayers = summary.LayerProfiles
            .Where(p => p.CheckpointCount > 10 && p.CacheHitRate < 0.5)
            .OrderBy(p => p.CacheHitRate)
            .Take(3)
            .ToList();

        foreach (var layer in lowCacheHitLayers)
        {
            recommendations.Add(new OptimizationRecommendation
            {
                Type = RecommendationType.ImproveCacheHitRate,
                Priority = RecommendationPriority.Medium,
                Title = $"Improve cache hit rate for layer '{layer.LayerId}'",
                Description = $"Layer '{layer.LayerId}' has a cache hit rate of only {layer.CacheHitRate:P0}. " +
                              $"Consider enabling recomputation cache or adjusting checkpoint strategy.",
                AffectedLayerId = layer.LayerId,
                ExpectedImpact = $"Could improve cache hit rate to ~{Math.Min(1.0, layer.CacheHitRate + 0.3):P0}"
            });
        }

        // Check for high memory consumption with low savings
        var lowEfficiencyLayers = summary.LayerProfiles
            .Where(p => p.TotalMemorySaved > 0 &&
                       p.TotalCheckpointTimeMs > p.TotalRecomputeTimeMs * 3)
            .OrderByDescending(p => p.TotalCheckpointTimeMs)
            .Take(3)
            .ToList();

        foreach (var layer in lowEfficiencyLayers)
        {
            recommendations.Add(new OptimizationRecommendation
            {
                Type = RecommendationType.ImproveEfficiency,
                Priority = RecommendationPriority.Low,
                Title = $"Improve checkpointing efficiency for layer '{layer.LayerId}'",
                Description = $"Layer '{layer.LayerId}' has high checkpoint overhead compared to savings. " +
                              $"Consider skipping checkpointing for this layer or using a different strategy.",
                AffectedLayerId = layer.LayerId,
                ExpectedImpact = $"Could reduce overhead by ~{(layer.TotalCheckpointTimeMs - layer.TotalRecomputeTimeMs) * 100 / layer.TotalCheckpointTimeMs:F0}%"
            });
        }

        return recommendations;
    }

    private string FormatMs(long ms)
    {
        if (ms < 1000) return $"{ms}ms";
        if (ms < 60000) return $"{ms / 1000.0:F1}s";
        return $"{ms / 60000.0:F1}m";
    }
}
