using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MLFramework.LoRA;
using MLFramework.Modules;
using MLFramework.NN;
using RitterFramework.Core;
using RitterFramework.Core.LoRA;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.LoRA
{
    /// <summary>
    /// Integration tests for LoRA functionality
    /// </summary>
    public class LoRAIntegrationTests : IDisposable
    {
        private readonly List<string> _tempFiles = new List<string>();

        public void Dispose()
        {
            foreach (var file in _tempFiles)
            {
                try
                {
                    if (File.Exists(file))
                        File.Delete(file);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }

        private string GetTempFilePath(string extension)
        {
            var path = Path.Combine(Path.GetTempPath(), $"test_lora_{Guid.NewGuid()}.{extension}");
            _tempFiles.Add(path);
            return path;
        }

        private class SimpleTransformer : IHierarchicalModule
        {
            public string Name { get; set; } = "simple_transformer";
            public IHierarchicalModule Parent { get; set; }

            public LinearWrapper QProj { get; }
            public LinearWrapper KProj { get; }
            public LinearWrapper VProj { get; }

            public SimpleTransformer()
            {
                var hiddenDim = 128;
                var ffDim = 512;

                QProj = new LinearWrapper(new Linear(hiddenDim, hiddenDim), "q_proj");
                KProj = new LinearWrapper(new Linear(hiddenDim, hiddenDim), "k_proj");
                VProj = new LinearWrapper(new Linear(hiddenDim, hiddenDim), "v_proj");
            }

            public IEnumerable<IHierarchicalModule> Children()
            {
                yield return QProj;
                yield return KProj;
                yield return VProj;
            }

            public void ReplaceChild(string name, IHierarchicalModule newChild)
            {
                if (name == QProj.Name) QProj = (LinearWrapper)newChild;
                else if (name == KProj.Name) KProj = (LinearWrapper)newChild;
                else if (name == VProj.Name) VProj = (LinearWrapper)newChild;
            }

            public Tensor Forward(Tensor input)
            {
                // Simple forward pass through attention projections
                var q = QProj.Forward(input);
                var k = KProj.Forward(input);
                var v = VProj.Forward(input);

                // Simplified attention: just return sum of projections
                return q + k + v;
            }
        }

        [Fact]
        public void EndToEnd_TrainWithLoRA()
        {
            // Arrange
            var model = new SimpleTransformer();
            var config = new LoraConfig { Rank = 4, Alpha = 8, TargetModules = new[] { "q_proj", "v_proj" } };

            // Inject LoRA
            LoraInjector.Inject(model, config);
            Assert.True(LoraInjector.HasLoRA(model));

            // Get trainable parameters (only LoRA matrices)
            var loraLayers = LoraInjector.GetLoRALayers(model);
            var trainableParams = loraLayers.SelectMany(l => l.TrainableParameters).ToList();

            Assert.NotEmpty(trainableParams);

            // Simulate forward pass
            var input = CreateRandomTensor(new[] { 2, 128 });
            var output = model.Forward(input);

            Assert.NotNull(output);
            Assert.Equal(new[] { 2, 128 }, output.Shape);

            // Verify LoRA parameters are trainable
            foreach (var param in trainableParams)
            {
                Assert.True(param.RequiresGrad);
            }
        }

        [Fact]
        public void EndToEnd_MergeAndUnmerge()
        {
            // Arrange
            var model = new SimpleTransformer();
            var config = new LoraConfig { Rank = 4, Alpha = 8 };
            LoraInjector.Inject(model, config);

            var input = CreateRandomTensor(new[] { 2, 128 });

            // Act - Get output before merge
            var outputBeforeMerge = model.Forward(input);

            // Merge LoRA
            var loraLayers = LoraInjector.GetLoRALayers(model);
            foreach (var layer in loraLayers)
            {
                layer.Merge();
            }

            // Get output after merge
            var outputAfterMerge = model.Forward(input);

            // Unmerge LoRA
            foreach (var layer in loraLayers)
            {
                layer.Unmerge();
            }

            // Get output after unmerge
            var outputAfterUnmerge = model.Forward(input);

            // Assert - All outputs should be identical
            Assert.Equal(outputBeforeMerge.Shape, outputAfterMerge.Shape);
            Assert.Equal(outputAfterMerge.Shape, outputAfterUnmerge.Shape);

            for (int i = 0; i < outputBeforeMerge.Size; i++)
            {
                Assert.Equal(outputBeforeMerge.Data[i], outputAfterMerge.Data[i], 5); // Allow small floating point differences
                Assert.Equal(outputAfterMerge.Data[i], outputAfterUnmerge.Data[i], 5);
            }
        }

        [Fact]
        public void EndToEnd_InjectAndRemove()
        {
            // Arrange
            var model = new SimpleTransformer();
            var input = CreateRandomTensor(new[] { 2, 128 });

            // Get baseline output
            var baselineOutput = model.Forward(input);

            // Act - Inject LoRA
            var config = new LoraConfig { Rank = 4, Alpha = 8 };
            LoraInjector.Inject(model, config);
            Assert.True(LoraInjector.HasLoRA(model));

            var loraOutput = model.Forward(input);

            // Remove LoRA
            LoraInjector.Remove(model);
            Assert.False(LoraInjector.HasLoRA(model));

            var afterRemoveOutput = model.Forward(input);

            // Assert
            Assert.NotNull(loraOutput);
            Assert.NotNull(afterRemoveOutput);

            // After removal, output should be similar to baseline
            for (int i = 0; i < baselineOutput.Size; i++)
            {
                Assert.Equal(baselineOutput.Data[i], afterRemoveOutput.Data[i], 5);
            }
        }

        [Fact]
        public void EndToEnd_CreateAdapterFromModel()
        {
            // Arrange
            var model = new SimpleTransformer();
            var config = new LoraConfig { Rank = 4, Alpha = 8, TargetModules = new[] { "q_proj" } };
            LoraInjector.Inject(model, config);

            // Act - Create adapter from model
            var loraLayers = LoraInjector.GetLoRALayers(model);
            var adapter = new LoraAdapter("test_adapter", config);

            foreach (var layer in loraLayers)
            {
                var (matrixA, matrixB) = layer.GetAdapterWeights();
                if (matrixA != null && matrixB != null)
                {
                    adapter.AddModuleWeights(layer.Name ?? "unknown", matrixA, matrixB);
                }
            }

            // Assert
            Assert.Single(adapter.Weights);
            Assert.Contains("q_proj", adapter.Weights.Keys);
        }

        [Fact]
        public void EndToEnd_LoadAdapterToModel()
        {
            // Arrange
            var sourceModel = new SimpleTransformer();
            var config = new LoraConfig { Rank = 4, Alpha = 8, TargetModules = new[] { "q_proj" } };
            LoraInjector.Inject(sourceModel, config);

            // Create adapter with some weights
            var loraLayers = LoraInjector.GetLoRALayers(sourceModel);
            var adapter = new LoraAdapter("test_adapter", config);

            foreach (var layer in loraLayers)
            {
                var (matrixA, matrixB) = layer.GetAdapterWeights();
                if (matrixA != null && matrixB != null)
                {
                    adapter.AddModuleWeights(layer.Name ?? "unknown", matrixA, matrixB);
                }
            }

            // Act - Load adapter to a new model
            var targetModel = new SimpleTransformer();
            LoraInjector.Inject(targetModel, config);
            var targetLayers = LoraInjector.GetLoRALayers(targetModel);

            // Apply adapter weights to target model
            foreach (var layer in targetLayers)
            {
                if (adapter.TryGetModuleWeights(layer.Name ?? "unknown", out var weights))
                {
                    layer.SetAdapterWeights(weights.LoraA, weights.LoraB);
                }
            }

            // Assert
            var input = CreateRandomTensor(new[] { 2, 128 });
            var output1 = sourceModel.Forward(input);
            var output2 = targetModel.Forward(input);

            Assert.NotNull(output2);
            Assert.Equal(output1.Shape, output2.Shape);
        }

        [Fact]
        public void EndToEnd_MultipleActiveAdapters()
        {
            // Arrange
            var model = new SimpleTransformer();
            var config = new LoraConfig { Rank = 4, Alpha = 8 };
            LoraInjector.Inject(model, config);

            var input = CreateRandomTensor(new[] { 2, 128 });
            var baselineOutput = model.Forward(input);

            // Create first adapter
            var loraLayers = LoraInjector.GetLoRALayers(model);
            var adapter1 = new LoraAdapter("adapter1", config);

            foreach (var layer in loraLayers)
            {
                var (matrixA, matrixB) = layer.GetAdapterWeights();
                if (matrixA != null && matrixB != null)
                {
                    // Modify weights slightly
                    var modifiedA = matrixA.Clone();
                    modifiedA.Data[0] += 0.1f;
                    adapter1.AddModuleWeights(layer.Name ?? "unknown", modifiedA, matrixB);
                }
            }

            // Create second adapter
            var adapter2 = new LoraAdapter("adapter2", config);
            foreach (var layer in loraLayers)
            {
                var (matrixA, matrixB) = layer.GetAdapterWeights();
                if (matrixA != null && matrixB != null)
                {
                    // Modify weights differently
                    var modifiedA = matrixA.Clone();
                    modifiedA.Data[0] -= 0.1f;
                    adapter2.AddModuleWeights(layer.Name ?? "unknown", modifiedA, matrixB);
                }
            }

            // Act - Apply both adapters
            var targetLayers = LoraInjector.GetLoRALayers(model);
            foreach (var layer in targetLayers)
            {
                if (adapter1.TryGetModuleWeights(layer.Name ?? "unknown", out var weights1))
                {
                    // Start with adapter1 weights
                    layer.SetAdapterWeights(weights1.LoraA, weights1.LoraB);
                }
            }

            var output1 = model.Forward(input);

            // Now apply adapter2 (simulating switching)
            foreach (var layer in targetLayers)
            {
                if (adapter2.TryGetModuleWeights(layer.Name ?? "unknown", out var weights2))
                {
                    layer.SetAdapterWeights(weights2.LoraA, weights2.LoraB);
                }
            }

            var output2 = model.Forward(input);

            // Assert
            Assert.NotEqual(output1.Data, output2.Data); // Outputs should differ
        }

        [Fact]
        public void EndToEnd_WithDifferentRanks()
        {
            // Arrange & Act
            var model1 = new SimpleTransformer();
            var config1 = new LoraConfig { Rank = 4, Alpha = 8 };
            LoraInjector.Inject(model1, config1);

            var model2 = new SimpleTransformer();
            var config2 = new LoraConfig { Rank = 16, Alpha = 32 };
            LoraInjector.Inject(model2, config2);

            var input = CreateRandomTensor(new[] { 2, 128 });
            var output1 = model1.Forward(input);
            var output2 = model2.Forward(input);

            // Assert
            Assert.NotNull(output1);
            Assert.NotNull(output2);
            Assert.Equal(output1.Shape, output2.Shape);
        }

        [Fact]
        public void EndToEnd_AdapterCloneIsIndependent()
        {
            // Arrange
            var adapter = new LoraAdapter("original", new LoraConfig { Rank = 8, Alpha = 16 });
            adapter.AddModuleWeights("q_proj", CreateRandomTensor(new[] { 128, 8 }), CreateRandomTensor(new[] { 8, 64 }));

            // Act
            var cloned = adapter.Clone();

            // Modify original
            var originalWeights = adapter.Weights["q_proj"];
            originalWeights.LoraA.Data[0] = 999.0f;

            // Assert
            Assert.NotEqual(999.0f, cloned.Weights["q_proj"].LoraA.Data[0]);
        }

        [Fact]
        public void EndToEnd_MultipleLayersWithDifferentTargets()
        {
            // Arrange
            var model = new SimpleTransformer();
            var config = new LoraConfig
            {
                Rank = 4,
                Alpha = 8,
                TargetModules = new[] { "q_proj", "k_proj", "v_proj" }
            };

            // Act
            LoraInjector.Inject(model, config);
            var loraLayers = LoraInjector.GetLoRALayers(model);

            // Assert
            Assert.Equal(3, loraLayers.Count);
            Assert.All(loraLayers, layer => Assert.IsType<LoraLinear>(layer));
        }

        [Fact]
        public void EndToEnd_LoRAParametersOnlyFractionOfBase()
        {
            // Arrange
            var hiddenDim = 128;
            var rank = 4;

            var model = new SimpleTransformer();
            var config = new LoraConfig { Rank = rank, Alpha = 8, TargetModules = new[] { "q_proj", "v_proj" } };

            // Calculate parameter counts
            // Base linear: 128 * 128 = 16,384 parameters per layer
            // LoRA: (128 * 4) + (4 * 128) = 1,024 parameters per layer
            var baseParamsPerLayer = hiddenDim * hiddenDim;
            var loraParamsPerLayer = (hiddenDim * rank) + (rank * hiddenDim);

            // Act
            LoraInjector.Inject(model, config);
            var loraLayers = LoraInjector.GetLoRALayers(model);
            var totalLoRAParams = loraLayers.Sum(l =>
                l.TrainableParameters.Sum(p => p.Size));

            var expectedLoRAParams = 2 * loraParamsPerLayer; // 2 layers

            // Assert
            Assert.Equal(expectedLoRAParams, totalLoRAParams);
            Assert.True(totalLoRAParams < baseParamsPerLayer); // LoRA should have fewer params than single base layer
        }

        [Fact]
        public void EndToEnd_LoRAWithDropout()
        {
            // Arrange
            var model = new SimpleTransformer();
            var config = new LoraConfig { Rank = 4, Alpha = 8, Dropout = 0.5f };
            LoraInjector.Inject(model, config);

            var input = CreateRandomTensor(new[] { 2, 128 });

            // Act
            var output1 = model.Forward(input);
            var output2 = model.Forward(input);

            // Assert
            Assert.NotNull(output1);
            Assert.NotNull(output2);
            // With dropout, outputs should differ (unless very unlucky)
            // Note: This test is probabilistic, but with dropout=0.5 and many elements,
            // it's extremely unlikely to get identical outputs
        }

        private Tensor CreateRandomTensor(int[] shape)
        {
            var random = new Random(42);
            var data = new float[shape.Aggregate(1, (x, y) => x * y)];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)random.NextDouble() * 2.0f - 1.0f; // [-1, 1]
            }
            return new Tensor(data, shape, false);
        }
    }
}
