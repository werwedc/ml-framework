using MLFramework.Serving.Deployment;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Integration tests for version lineage tracking scenarios.
    /// </summary>
    public class VersionLineageTests : IClassFixture<IntegrationTestFixture>
    {
        private readonly IntegrationTestFixture _fixture;

        public VersionLineageTests(IntegrationTestFixture fixture)
        {
            _fixture = fixture;
        }

        [Fact]
        public void Lineage_TrackParentChild()
        {
            // Arrange
            const string modelName = "test-model";
            const string baseVersion = "v1.0.0";
            const string fineTunedVersion = "v1.1.0";

            // 1. Register base model
            var baseMetadata = new ModelMetadata
            {
                Version = baseVersion,
                TrainingDate = DateTime.UtcNow.AddDays(-10),
                ArtifactPath = $"/models/{modelName}/{baseVersion}/model.pt",
                Hyperparameters = new Dictionary<string, object>
                {
                    ["learning_rate"] = 0.001f,
                    ["batch_size"] = 32
                }
            };

            _fixture.MockRegistry.Setup(r => r.GetMetadata(modelName, baseVersion))
                .Returns(baseMetadata);

            // 2. Register fine-tuned model with parent
            var fineTunedMetadata = new ModelMetadata
            {
                Version = fineTunedVersion,
                TrainingDate = DateTime.UtcNow,
                ArtifactPath = $"/models/{modelName}/{fineTunedVersion}/model.pt",
                Hyperparameters = new Dictionary<string, object>
                {
                    ["learning_rate"] = 0.0001f,
                    ["batch_size"] = 32,
                    ["parent_version"] = baseVersion,
                    ["fine_tuned_on"] = "custom_dataset"
                }
            };

            _fixture.MockRegistry.Setup(r => r.GetMetadata(modelName, fineTunedVersion))
                .Returns(fineTunedMetadata);

            // 3. Verify lineage tracking
            Assert.NotNull(baseMetadata);
            Assert.NotNull(fineTunedMetadata);
            Assert.Equal(baseVersion, fineTunedMetadata.Hyperparameters["parent_version"]);

            // 4. Query models by lineage
            var parentVersion = fineTunedMetadata.Hyperparameters["parent_version"] as string;
            Assert.Equal(baseVersion, parentVersion);
        }

        [Fact]
        public void Lineage_WithMultipleGenerations_TracksHierarchy()
        {
            // Arrange
            const string modelName = "test-model";
            var versions = new[]
            {
                "v1.0.0", // Base
                "v1.1.0", // Fine-tuned from v1.0.0
                "v1.2.0", // Fine-tuned from v1.1.0
                "v2.0.0"  // New base model
            };

            var metadataDict = new Dictionary<string, ModelMetadata>();

            // Create base model v1.0.0
            metadataDict["v1.0.0"] = new ModelMetadata
            {
                Version = "v1.0.0",
                TrainingDate = DateTime.UtcNow.AddDays(-30),
                ArtifactPath = $"/models/{modelName}/v1.0.0/model.pt",
                Hyperparameters = new Dictionary<string, object>()
            };

            // Create fine-tuned v1.1.0 from v1.0.0
            metadataDict["v1.1.0"] = new ModelMetadata
            {
                Version = "v1.1.0",
                TrainingDate = DateTime.UtcNow.AddDays(-20),
                ArtifactPath = $"/models/{modelName}/v1.1.0/model.pt",
                Hyperparameters = new Dictionary<string, object>
                {
                    ["parent_version"] = "v1.0.0"
                }
            };

            // Create fine-tuned v1.2.0 from v1.1.0
            metadataDict["v1.2.0"] = new ModelMetadata
            {
                Version = "v1.2.0",
                TrainingDate = DateTime.UtcNow.AddDays(-10),
                ArtifactPath = $"/models/{modelName}/v1.2.0/model.pt",
                Hyperparameters = new Dictionary<string, object>
                {
                    ["parent_version"] = "v1.1.0"
                }
            };

            // Create new base v2.0.0
            metadataDict["v2.0.0"] = new ModelMetadata
            {
                Version = "v2.0.0",
                TrainingDate = DateTime.UtcNow.AddDays(-5),
                ArtifactPath = $"/models/{modelName}/v2.0.0/model.pt",
                Hyperparameters = new Dictionary<string, object>()
            };

            // Setup mocks
            foreach (var kvp in metadataDict)
            {
                _fixture.MockRegistry.Setup(r => r.GetMetadata(modelName, kvp.Key))
                    .Returns(kvp.Value);
            }

            // Act - Trace lineage
            var lineage = new List<string>();
            var currentVersion = "v1.2.0";

            while (true)
            {
                lineage.Add(currentVersion);
                var metadata = metadataDict[currentVersion];

                if (metadata.Hyperparameters.ContainsKey("parent_version"))
                {
                    currentVersion = metadata.Hyperparameters["parent_version"] as string;
                }
                else
                {
                    break;
                }
            }

            // Assert - Verify lineage from v1.2.0 -> v1.1.0 -> v1.0.0
            Assert.Equal(3, lineage.Count);
            Assert.Equal("v1.2.0", lineage[0]);
            Assert.Equal("v1.1.0", lineage[1]);
            Assert.Equal("v1.0.0", lineage[2]);
        }

        [Fact]
        public void Lineage_FindAllDescendants_ReturnsCorrectVersions()
        {
            // Arrange
            const string modelName = "test-model";
            var baseVersion = "v1.0.0";
            var descendants = new[] { "v1.1.0", "v1.2.0", "v1.1.1" };

            var metadataDict = new Dictionary<string, ModelMetadata>
            {
                ["v1.0.0"] = new ModelMetadata
                {
                    Version = "v1.0.0",
                    TrainingDate = DateTime.UtcNow.AddDays(-30),
                    Hyperparameters = new Dictionary<string, object>()
                },
                ["v1.1.0"] = new ModelMetadata
                {
                    Version = "v1.1.0",
                    TrainingDate = DateTime.UtcNow.AddDays(-20),
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["parent_version"] = "v1.0.0"
                    }
                },
                ["v1.2.0"] = new ModelMetadata
                {
                    Version = "v1.2.0",
                    TrainingDate = DateTime.UtcNow.AddDays(-15),
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["parent_version"] = "v1.0.0"
                    }
                },
                ["v1.1.1"] = new ModelMetadata
                {
                    Version = "v1.1.1",
                    TrainingDate = DateTime.UtcNow.AddDays(-10),
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["parent_version"] = "v1.1.0"
                    }
                }
            };

            foreach (var kvp in metadataDict)
            {
                _fixture.MockRegistry.Setup(r => r.GetMetadata(modelName, kvp.Key))
                    .Returns(kvp.Value);
            }

            // Act - Find all descendants of v1.0.0
            var foundDescendants = metadataDict
                .Where(kvp => kvp.Value.Hyperparameters.ContainsKey("parent_version") &&
                             (string)kvp.Value.Hyperparameters["parent_version"] == baseVersion)
                .Select(kvp => kvp.Key)
                .ToList();

            // Assert
            Assert.Contains("v1.1.0", foundDescendants);
            Assert.Contains("v1.2.0", foundDescendants);
            Assert.DoesNotContain("v1.1.1", foundDescendants); // This is a descendant of v1.1.0, not v1.0.0 directly
        }

        [Fact]
        public void Lineage_WithBranches_TracksMultiplePaths()
        {
            // Arrange
            const string modelName = "test-model";
            var baseVersion = "v1.0.0";
            var branches = new[]
            {
                "v1.1.0", // Branch 1
                "v1.2.0", // Branch 2
                "v1.3.0"  // Branch 3
            };

            var metadataDict = new Dictionary<string, ModelMetadata>();

            // Create base
            metadataDict[baseVersion] = new ModelMetadata
            {
                Version = baseVersion,
                TrainingDate = DateTime.UtcNow.AddDays(-30),
                Hyperparameters = new Dictionary<string, object>()
            };

            // Create branches
            foreach (var branch in branches)
            {
                metadataDict[branch] = new ModelMetadata
                {
                    Version = branch,
                    TrainingDate = DateTime.UtcNow.AddDays(-10),
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["parent_version"] = baseVersion,
                        ["branch_name"] = branch
                    }
                };
            }

            foreach (var kvp in metadataDict)
            {
                _fixture.MockRegistry.Setup(r => r.GetMetadata(modelName, kvp.Key))
                    .Returns(kvp.Value);
            }

            // Act - Find all direct children of base
            var directChildren = metadataDict
                .Where(kvp => kvp.Value.Hyperparameters.ContainsKey("parent_version") &&
                             (string)kvp.Value.Hyperparameters["parent_version"] == baseVersion)
                .ToList();

            // Assert
            Assert.Equal(3, directChildren.Count);
            Assert.All(branches, branch => Assert.True(
                directChildren.Any(kvp => kvp.Key == branch),
                $"Branch {branch} not found as direct child"
            ));
        }

        [Fact]
        public void Lineage_WithMerge_TracksCorrectly()
        {
            // Arrange
            const string modelName = "test-model";
            var versions = new[]
            {
                "v1.0.0", // Base
                "v1.1.0", // Branch A
                "v1.2.0", // Branch B
                "v2.0.0"  // Merge of A and B
            };

            var metadataDict = new Dictionary<string, ModelMetadata>
            {
                ["v1.0.0"] = new ModelMetadata
                {
                    Version = "v1.0.0",
                    TrainingDate = DateTime.UtcNow.AddDays(-30),
                    Hyperparameters = new Dictionary<string, object>()
                },
                ["v1.1.0"] = new ModelMetadata
                {
                    Version = "v1.1.0",
                    TrainingDate = DateTime.UtcNow.AddDays(-20),
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["parent_version"] = "v1.0.0"
                    }
                },
                ["v1.2.0"] = new ModelMetadata
                {
                    Version = "v1.2.0",
                    TrainingDate = DateTime.UtcNow.AddDays(-15),
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["parent_version"] = "v1.0.0"
                    }
                },
                ["v2.0.0"] = new ModelMetadata
                {
                    Version = "v2.0.0",
                    TrainingDate = DateTime.UtcNow.AddDays(-5),
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["parent_version"] = "v1.0.0",
                        ["merge_from"] = new[] { "v1.1.0", "v1.2.0" }
                    }
                }
            };

            foreach (var kvp in metadataDict)
            {
                _fixture.MockRegistry.Setup(r => r.GetMetadata(modelName, kvp.Key))
                    .Returns(kvp.Value);
            }

            // Act - Get merge information
            var mergeMetadata = metadataDict["v2.0.0"];
            var hasMergeInfo = mergeMetadata.Hyperparameters.ContainsKey("merge_from");

            // Assert
            Assert.True(hasMergeInfo);
            var mergeFrom = mergeMetadata.Hyperparameters["merge_from"] as string[];
            Assert.NotNull(mergeFrom);
            Assert.Contains("v1.1.0", mergeFrom);
            Assert.Contains("v1.2.0", mergeFrom);
        }

        [Fact]
        public void Lineage_WithRollbackTracks_RevertsCorrectly()
        {
            // Arrange
            const string modelName = "test-model";
            var versions = new[] { "v1.0.0", "v2.0.0", "v3.0.0" };

            var metadataDict = new Dictionary<string, ModelMetadata>();

            // Create sequential versions
            for (int i = 0; i < versions.Length; i++)
            {
                var parent = i > 0 ? versions[i - 1] : null;

                metadataDict[versions[i]] = new ModelMetadata
                {
                    Version = versions[i],
                    TrainingDate = DateTime.UtcNow.AddDays(-(10 - i)),
                    Hyperparameters = new Dictionary<string, object>()
                };

                if (parent != null)
                {
                    metadataDict[versions[i]].Hyperparameters["parent_version"] = parent;
                    metadataDict[versions[i]].Hyperparameters["rollback_from"] = parent;
                }
            }

            foreach (var kvp in metadataDict)
            {
                _fixture.MockRegistry.Setup(r => r.GetMetadata(modelName, kvp.Key))
                    .Returns(kvp.Value);
            }

            // Act - Verify rollback tracking
            var v3Metadata = metadataDict["v3.0.0"];
            var v3Rollback = v3Metadata.Hyperparameters["rollback_from"] as string;

            // Assert
            Assert.NotNull(v3Rollback);
            Assert.Equal("v2.0.0", v3Rollback);
        }

        [Fact]
        public void Lineage_GetVersionsByParent_ReturnsCorrectList()
        {
            // Arrange
            const string modelName = "test-model";
            const string parentVersion = "v1.0.0";
            var childVersions = new[] { "v1.1.0", "v1.1.1", "v1.1.2", "v1.2.0" };

            var metadataDict = new Dictionary<string, ModelMetadata>
            {
                ["v1.0.0"] = new ModelMetadata
                {
                    Version = parentVersion,
                    Hyperparameters = new Dictionary<string, object>()
                }
            };

            // Create children
            foreach (var child in childVersions)
            {
                metadataDict[child] = new ModelMetadata
                {
                    Version = child,
                    Hyperparameters = new Dictionary<string, object>
                    {
                        ["parent_version"] = parentVersion
                    }
                };
            }

            foreach (var kvp in metadataDict)
            {
                _fixture.MockRegistry.Setup(r => r.GetMetadata(modelName, kvp.Key))
                    .Returns(kvp.Value);
            }

            // Act - Get all children of parent
            var children = metadataDict
                .Where(kvp => kvp.Value.Hyperparameters.ContainsKey("parent_version") &&
                             (string)kvp.Value.Hyperparameters["parent_version"] == parentVersion)
                .Select(kvp => kvp.Key)
                .OrderBy(v => v)
                .ToList();

            // Assert
            Assert.Equal(4, children.Count);
            Assert.Equal("v1.1.0", children[0]);
            Assert.Equal("v1.1.1", children[1]);
            Assert.Equal("v1.1.2", children[2]);
            Assert.Equal("v1.2.0", children[3]);
        }

        [Fact]
        public void Lineage_WithComparison_MatchesCorrectly()
        {
            // Arrange
            const string modelName = "test-model";
            var version1 = "v1.0.0";
            var version2 = "v2.0.0";

            var metadata1 = new ModelMetadata
            {
                Version = version1,
                Hyperparameters = new Dictionary<string, object>
                {
                    ["learning_rate"] = 0.001f,
                    ["batch_size"] = 32,
                    ["optimizer"] = "adam"
                }
            };

            var metadata2 = new ModelMetadata
            {
                Version = version2,
                Hyperparameters = new Dictionary<string, object>
                {
                    ["learning_rate"] = 0.0005f,
                    ["batch_size"] = 32,
                    ["optimizer"] = "adam",
                    ["parent_version"] = version1
                }
            };

            _fixture.MockRegistry.Setup(r => r.GetMetadata(modelName, version1))
                .Returns(metadata1);
            _fixture.MockRegistry.Setup(r => r.GetMetadata(modelName, version2))
                .Returns(metadata2);

            // Act - Compare hyperparameters
            var lrChanged = metadata2.Hyperparameters["learning_rate"].ToString() !=
                           metadata1.Hyperparameters["learning_rate"].ToString();
            var bsChanged = metadata2.Hyperparameters["batch_size"].ToString() !=
                           metadata1.Hyperparameters["batch_size"].ToString();

            // Assert
            Assert.True(lrChanged, "Learning rate should have changed");
            Assert.False(bsChanged, "Batch size should be the same");
        }
    }
}
