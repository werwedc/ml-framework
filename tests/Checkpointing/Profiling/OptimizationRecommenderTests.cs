using MLFramework.Checkpointing.Profiling;

namespace MLFramework.Tests.Checkpointing.Profiling;

public class OptimizationRecommenderTests
{
    private readonly OptimizationRecommender _recommender;

    public OptimizationRecommenderTests()
    {
        _recommender = new OptimizationRecommender();
    }

    [Fact]
    public void GenerateRecommendations_RecommendsForHighRecomputeLayers()
    {
        // Arrange
        var summary = new ProfilingSummary
        {
            StartTime = DateTime.UtcNow,
            EndTime = DateTime.UtcNow,
            Duration = 0,
            TotalEvents = 0,
            LayerProfiles = new List<LayerProfile>
            {
                new LayerProfile
                {
                    LayerId = "high_recompute",
                    CheckpointCount = 1,
                    RecomputeCount = 10,
                    TotalRecomputeTimeMs = 500,
                    TotalCheckpointTimeMs = 100,
                    TotalMemorySaved = 1024
                }
            }
        };

        // Act
        var recommendations = _recommender.GenerateRecommendations(summary);

        // Assert
        var highRecomputeRec = recommendations.FirstOrDefault(r =>
            r.Type == RecommendationType.ReduceRecomputation &&
            r.AffectedLayerId == "high_recompute");

        Assert.NotNull(highRecomputeRec);
        Assert.Equal(RecommendationPriority.High, highRecomputeRec.Priority);
        Assert.Contains("Reduce recomputation", highRecomputeRec.Title);
    }

    [Fact]
    public void GenerateRecommendations_RecommendsForLowCacheHitLayers()
    {
        // Arrange
        var summary = new ProfilingSummary
        {
            StartTime = DateTime.UtcNow,
            EndTime = DateTime.UtcNow,
            Duration = 0,
            TotalEvents = 0,
            LayerProfiles = new List<LayerProfile>
            {
                new LayerProfile
                {
                    LayerId = "low_cache_hit",
                    CheckpointCount = 20,
                    RecomputeCount = 10,
                    CacheHitCount = 5,
                    TotalRecomputeTimeMs = 100,
                    TotalCheckpointTimeMs = 200,
                    TotalMemorySaved = 1024
                }
            }
        };

        // Act
        var recommendations = _recommender.GenerateRecommendations(summary);

        // Assert
        var lowCacheHitRec = recommendations.FirstOrDefault(r =>
            r.Type == RecommendationType.ImproveCacheHitRate &&
            r.AffectedLayerId == "low_cache_hit");

        Assert.NotNull(lowCacheHitRec);
        Assert.Equal(RecommendationPriority.Medium, lowCacheHitRec.Priority);
        Assert.Contains("Improve cache hit rate", lowCacheHitRec.Title);
    }

    [Fact]
    public void GenerateRecommendations_RecommendsForLowEfficiencyLayers()
    {
        // Arrange
        var summary = new ProfilingSummary
        {
            StartTime = DateTime.UtcNow,
            EndTime = DateTime.UtcNow,
            Duration = 0,
            TotalEvents = 0,
            LayerProfiles = new List<LayerProfile>
            {
                new LayerProfile
                {
                    LayerId = "low_efficiency",
                    CheckpointCount = 10,
                    RecomputeCount = 1,
                    TotalRecomputeTimeMs = 100,
                    TotalCheckpointTimeMs = 500,
                    TotalMemorySaved = 1024
                }
            }
        };

        // Act
        var recommendations = _recommender.GenerateRecommendations(summary);

        // Assert
        var lowEfficiencyRec = recommendations.FirstOrDefault(r =>
            r.Type == RecommendationType.ImproveEfficiency &&
            r.AffectedLayerId == "low_efficiency");

        Assert.NotNull(lowEfficiencyRec);
        Assert.Equal(RecommendationPriority.Low, lowEfficiencyRec.Priority);
        Assert.Contains("Improve checkpointing efficiency", lowEfficiencyRec.Title);
    }

    [Fact]
    public void GenerateRecommendations_LimitsToTop3ForEachType()
    {
        // Arrange
        var summary = new ProfilingSummary
        {
            StartTime = DateTime.UtcNow,
            EndTime = DateTime.UtcNow,
            Duration = 0,
            TotalEvents = 0,
            LayerProfiles = new List<LayerProfile>
            {
                new LayerProfile { LayerId = "layer1", CheckpointCount = 1, RecomputeCount = 10, TotalRecomputeTimeMs = 500, TotalCheckpointTimeMs = 100, TotalMemorySaved = 1024 },
                new LayerProfile { LayerId = "layer2", CheckpointCount = 1, RecomputeCount = 10, TotalRecomputeTimeMs = 400, TotalCheckpointTimeMs = 100, TotalMemorySaved = 1024 },
                new LayerProfile { LayerId = "layer3", CheckpointCount = 1, RecomputeCount = 10, TotalRecomputeTimeMs = 300, TotalCheckpointTimeMs = 100, TotalMemorySaved = 1024 },
                new LayerProfile { LayerId = "layer4", CheckpointCount = 1, RecomputeCount = 10, TotalRecomputeTimeMs = 200, TotalCheckpointTimeMs = 100, TotalMemorySaved = 1024 },
                new LayerProfile { LayerId = "layer5", CheckpointCount = 1, RecomputeCount = 10, TotalRecomputeTimeMs = 100, TotalCheckpointTimeMs = 100, TotalMemorySaved = 1024 }
            }
        };

        // Act
        var recommendations = _recommender.GenerateRecommendations(summary);

        // Assert
        var recomputeRecs = recommendations.Where(r => r.Type == RecommendationType.ReduceRecomputation).ToList();
        Assert.Equal(3, recomputeRecs.Count);
    }

    [Fact]
    public void GenerateRecommendations_PrioritizesCorrectly()
    {
        // Arrange
        var summary = new ProfilingSummary
        {
            StartTime = DateTime.UtcNow,
            EndTime = DateTime.UtcNow,
            Duration = 0,
            TotalEvents = 0,
            LayerProfiles = new List<LayerProfile>
            {
                new LayerProfile
                {
                    LayerId = "high_priority",
                    CheckpointCount = 1,
                    RecomputeCount = 10,
                    TotalRecomputeTimeMs = 500,
                    TotalCheckpointTimeMs = 100,
                    TotalMemorySaved = 1024
                },
                new LayerProfile
                {
                    LayerId = "medium_priority",
                    CheckpointCount = 20,
                    RecomputeCount = 10,
                    CacheHitCount = 5,
                    TotalRecomputeTimeMs = 100,
                    TotalCheckpointTimeMs = 200,
                    TotalMemorySaved = 1024
                },
                new LayerProfile
                {
                    LayerId = "low_priority",
                    CheckpointCount = 10,
                    RecomputeCount = 1,
                    TotalRecomputeTimeMs = 100,
                    TotalCheckpointTimeMs = 500,
                    TotalMemorySaved = 1024
                }
            }
        };

        // Act
        var recommendations = _recommender.GenerateRecommendations(summary);

        // Assert
        Assert.Equal(RecommendationPriority.High, recommendations.First(r => r.AffectedLayerId == "high_priority").Priority);
        Assert.Equal(RecommendationPriority.Medium, recommendations.First(r => r.AffectedLayerId == "medium_priority").Priority);
        Assert.Equal(RecommendationPriority.Low, recommendations.First(r => r.AffectedLayerId == "low_priority").Priority);
    }

    [Fact]
    public void GenerateRecommendations_ReturnsEmptyListForNoIssues()
    {
        // Arrange
        var summary = new ProfilingSummary
        {
            StartTime = DateTime.UtcNow,
            EndTime = DateTime.UtcNow,
            Duration = 0,
            TotalEvents = 0,
            LayerProfiles = new List<LayerProfile>
            {
                new LayerProfile
                {
                    LayerId = "good_layer",
                    CheckpointCount = 1,
                    RecomputeCount = 1,
                    TotalRecomputeTimeMs = 100,
                    TotalCheckpointTimeMs = 100,
                    TotalMemorySaved = 1024,
                    CacheHitCount = 1
                }
            }
        };

        // Act
        var recommendations = _recommender.GenerateRecommendations(summary);

        // Assert
        Assert.Empty(recommendations);
    }
}
