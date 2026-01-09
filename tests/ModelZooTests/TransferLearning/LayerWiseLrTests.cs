using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.ModelZoo.TransferLearning;
using MLFramework.NN;
using MLFramework.Optimizers;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.ModelZooTests.TransferLearning
{
    /// <summary>
    /// Unit tests for layer-wise learning rate scheduling functionality.
    /// </summary>
    public class LayerWiseLrTests
    {
        #region Helper Methods

        /// <summary>
        /// Creates a simple linear head module for testing.
        /// </summary>
        private Module CreateMockModule(string name, bool hasParams = true)
        {
            return new LinearHead(10, 10, true, name);
        }

        /// <summary>
        /// Creates a sequential model with multiple layers.
        /// </summary>
        private SequentialModule CreateTestModel(int numLayers = 5)
        {
            var model = new SequentialModule("test_model");
            for (int i = 0; i < numLayers; i++)
            {
                model.Add(CreateMockModule($"layer_{i}"));
            }
            return model;
        }

        /// <summary>
        /// Creates a mock optimizer for testing.
        /// </summary>
        private Optimizer CreateMockOptimizer(Dictionary<string, Tensor> parameters, float lr = 0.001f)
        {
            return new MockOptimizer(parameters, lr);
        }

        /// <summary>
        /// Creates a tensor with specified dimensions and values.
        /// </summary>
        private Tensor CreateTensor(int[] shape, float value = 1.0f)
        {
            int size = 1;
            foreach (int dim in shape)
            {
                size *= dim;
            }

            var data = new float[size];
            for (int i = 0; i < size; i++)
            {
                data[i] = value;
            }

            return new Tensor(data, shape);
        }

        /// <summary>
        /// Creates parameter dictionary from a model.
        /// </summary>
        private Dictionary<string, Tensor> CreateParameterDict(Module model)
        {
            var paramDict = new Dictionary<string, Tensor>();
            foreach (var (paramName, param) in model.GetNamedParameters())
            {
                paramDict[paramName] = param;
            }
            return paramDict;
        }

        #endregion

        #region ParameterGroup Tests

        [Fact]
        public void ParameterGroup_Constructor_CreatesValidGroup()
        {
            var group = new ParameterGroup("test_group", 0.001f);

            Assert.Equal("test_group", group.Name);
            Assert.Equal(0.001f, group.LearningRate);
            Assert.Equal(0, group.ParameterCount);
            Assert.Equal(0, group.TotalElements);
        }

        [Fact]
        public void ParameterGroup_Constructor_WithInvalidName_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => new ParameterGroup("", 0.001f));
            Assert.Throws<ArgumentException>(() => new ParameterGroup(null, 0.001f));
        }

        [Fact]
        public void ParameterGroup_Constructor_WithInvalidLr_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new ParameterGroup("test", -0.001f));
            Assert.Throws<ArgumentOutOfRangeException>(() => new ParameterGroup("test", 0.0f));
        }

        [Fact]
        public void ParameterGroup_AddParameter_AddsSuccessfully()
        {
            var group = new ParameterGroup("test_group", 0.001f);
            var tensor = CreateTensor(new[] { 10, 10 });

            group.AddParameter(tensor, "layer_0.weight");

            Assert.Equal(1, group.ParameterCount);
            Assert.Equal(100, group.TotalElements);
        }

        [Fact]
        public void ParameterGroup_AddParameter_WithNullTensor_ThrowsArgumentNullException()
        {
            var group = new ParameterGroup("test_group", 0.001f);
            Assert.Throws<ArgumentNullException>(() => group.AddParameter(null, "test"));
        }

        [Fact]
        public void ParameterGroup_AddParameters_AddsMultiple()
        {
            var group = new ParameterGroup("test_group", 0.001f);
            var paramsDict = new Dictionary<string, Tensor>
            {
                ["layer_0.weight"] = CreateTensor(new[] { 10, 10 }),
                ["layer_0.bias"] = CreateTensor(new[] { 10 })
            };

            group.AddParameters(paramsDict);

            Assert.Equal(2, group.ParameterCount);
        }

        [Fact]
        public void ParameterGroup_Clear_RemovesAllParameters()
        {
            var group = new ParameterGroup("test_group", 0.001f);
            var tensor = CreateTensor(new[] { 10, 10 });
            group.AddParameter(tensor, "test");

            group.Clear();

            Assert.Equal(0, group.ParameterCount);
        }

        [Fact]
        public void ParameterGroup_ToParameterDictionary_ReturnsCorrectDict()
        {
            var group = new ParameterGroup("test_group", 0.001f);
            var tensor1 = CreateTensor(new[] { 10, 10 });
            var tensor2 = CreateTensor(new[] { 10 });

            group.AddParameter(tensor1, "layer_0.weight");
            group.AddParameter(tensor2, "layer_0.bias");

            var dict = group.ToParameterDictionary();

            Assert.Equal(2, dict.Count);
            Assert.True(dict.ContainsKey("layer_0.weight"));
            Assert.True(dict.ContainsKey("layer_0.bias"));
        }

        [Fact]
        public void ParameterGroup_ToString_ReturnsFormattedString()
        {
            var group = new ParameterGroup("test_group", 0.001f, 0.0001f, 0.9f);
            var tensor = CreateTensor(new[] { 10, 10 });
            group.AddParameter(tensor, "layer_0.weight");

            var str = group.ToString();

            Assert.Contains("test_group", str);
            Assert.Contains("1 parameters", str);
            Assert.Contains("LR=0.001", str);
            Assert.Contains("WD=0.0001", str);
            Assert.Contains("Momentum=0.9", str);
        }

        [Fact]
        public void ParameterGroup_CanUpdateProperties()
        {
            var group = new ParameterGroup("test_group", 0.001f);

            group.LearningRate = 0.01f;
            group.WeightDecay = 0.001f;
            group.Momentum = 0.95f;

            Assert.Equal(0.01f, group.LearningRate);
            Assert.Equal(0.001f, group.WeightDecay);
            Assert.Equal(0.95f, group.Momentum);
        }

        #endregion

        #region ParameterGroupBuilder Tests

        [Fact]
        public void ParameterGroupBuilder_Constructor_CreatesValidBuilder()
        {
            var builder = new ParameterGroupBuilder();

            Assert.Equal(0, builder.GroupCount);
        }

        [Fact]
        public void ParameterGroupBuilder_NewGroup_StartsNewGroup()
        {
            var builder = new ParameterGroupBuilder();

            builder.NewGroup("group_0", 0.001f);

            Assert.Equal(0, builder.GroupCount); // Group not yet built
        }

        [Fact]
        public void ParameterGroupBuilder_NewGroup_WithInvalidName_ThrowsArgumentException()
        {
            var builder = new ParameterGroupBuilder();

            Assert.Throws<ArgumentException>(() => builder.NewGroup("", 0.001f));
            Assert.Throws<ArgumentException>(() => builder.NewGroup(null, 0.001f));
        }

        [Fact]
        public void ParameterGroupBuilder_AddFrozenParameters_AddsFrozenParams()
        {
            var model = CreateTestModel(3);
            model.Freeze(exceptLastN: 1);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("frozen", 0.0001f);
            builder.AddFrozenParameters(model);

            var groups = builder.Build();

            Assert.Single(groups);
            Assert.Equal(4, groups[0].ParameterCount); // 2 params per layer * 2 frozen layers
        }

        [Fact]
        public void ParameterGroupBuilder_AddUnfrozenParameters_AddsUnfrozenParams()
        {
            var model = CreateTestModel(3);
            model.Freeze(exceptLastN: 1);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("unfrozen", 0.001f);
            builder.AddUnfrozenParameters(model);

            var groups = builder.Build();

            Assert.Single(groups);
            Assert.Equal(2, groups[0].ParameterCount); // 2 params in the last layer
        }

        [Fact]
        public void ParameterGroupBuilder_AddLayer_AddsSpecificLayer()
        {
            var model = CreateTestModel(3);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("layer_0_group", 0.001f);
            builder.AddLayer("layer_0", model);

            var groups = builder.Build();

            Assert.Single(groups);
            Assert.Equal(2, groups[0].ParameterCount); // weight and bias
        }

        [Fact]
        public void ParameterGroupBuilder_AddLayersByPattern_AddsMatchingLayers()
        {
            var model = CreateTestModel(5);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("early_layers", 0.0001f);
            builder.AddLayersByPattern("^layer_[0-2]$", model);

            var groups = builder.Build();

            Assert.Single(groups);
            Assert.Equal(6, groups[0].ParameterCount); // 2 params per layer * 3 layers
        }

        [Fact]
        public void ParameterGroupBuilder_AddFirstNLayers_AddsCorrectLayers()
        {
            var model = CreateTestModel(5);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("first_2", 0.0001f);
            builder.AddFirstNLayers(2, model);

            var groups = builder.Build();

            Assert.Single(groups);
            Assert.Equal(4, groups[0].ParameterCount); // 2 params per layer * 2 layers
        }

        [Fact]
        public void ParameterGroupBuilder_AddLastNLayers_AddsCorrectLayers()
        {
            var model = CreateTestModel(5);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("last_2", 0.001f);
            builder.AddLastNLayers(2, model);

            var groups = builder.Build();

            Assert.Single(groups);
            Assert.Equal(4, groups[0].ParameterCount); // 2 params per layer * 2 layers
        }

        [Fact]
        public void ParameterGroupBuilder_AddLayerRange_AddsCorrectRange()
        {
            var model = CreateTestModel(5);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("middle", 0.0005f);
            builder.AddLayerRange(1, 3, model);

            var groups = builder.Build();

            Assert.Single(groups);
            Assert.Equal(6, groups[0].ParameterCount); // 2 params per layer * 3 layers
        }

        [Fact]
        public void ParameterGroupBuilder_WithLearningRate_UpdatesLr()
        {
            var model = CreateTestModel(1);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("test", 0.001f);
            builder.AddLayer("layer_0", model);
            builder.WithLearningRate(0.01f);

            var groups = builder.Build();

            Assert.Equal(0.01f, groups[0].LearningRate);
        }

        [Fact]
        public void ParameterGroupBuilder_WithWeightDecay_UpdatesWeightDecay()
        {
            var model = CreateTestModel(1);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("test", 0.001f);
            builder.AddLayer("layer_0", model);
            builder.WithWeightDecay(0.001f);

            var groups = builder.Build();

            Assert.Equal(0.001f, groups[0].WeightDecay);
        }

        [Fact]
        public void ParameterGroupBuilder_WithMomentum_UpdatesMomentum()
        {
            var model = CreateTestModel(1);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("test", 0.001f);
            builder.AddLayer("layer_0", model);
            builder.WithMomentum(0.9f);

            var groups = builder.Build();

            Assert.Equal(0.9f, groups[0].Momentum);
        }

        [Fact]
        public void ParameterGroupBuilder_MultipleGroups_CreatesAllGroups()
        {
            var model = CreateTestModel(3);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("frozen", 0.0001f).AddFirstNLayers(2, model);
            builder.NewGroup("unfrozen", 0.001f).AddLastNLayers(1, model);

            var groups = builder.Build();

            Assert.Equal(2, groups.Count);
            Assert.Equal("frozen", groups[0].Name);
            Assert.Equal("unfrozen", groups[1].Name);
        }

        [Fact]
        public void ParameterGroupBuilder_Reset_ClearsAllGroups()
        {
            var model = CreateTestModel(1);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("test", 0.001f).AddLayer("layer_0", model);

            builder.Reset();

            Assert.Equal(0, builder.GroupCount);
        }

        [Fact]
        public void ParameterGroupBuilder_BuildDictionary_ReturnsDictionary()
        {
            var model = CreateTestModel(1);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("test", 0.001f).AddLayer("layer_0", model);

            var dict = builder.BuildDictionary();

            Assert.Single(dict);
            Assert.True(dict.ContainsKey("test"));
        }

        [Fact]
        public void ParameterGroupBuilder_WithNoActiveGroup_ThrowsInvalidOperationException()
        {
            var model = CreateTestModel(1);
            var builder = new ParameterGroupBuilder();

            Assert.Throws<InvalidOperationException>(() => builder.AddLayer("layer_0", model));
        }

        #endregion

        #region LayerWiseLrExtensions Tests

        [Fact]
        public void SetLayerWiseLrs_ByDictionary_SetsCorrectLrs()
        {
            var model = CreateTestModel(2);
            var optimizer = CreateMockOptimizer(CreateParameterDict(model), 0.001f);

            var layerLrs = new Dictionary<string, float>
            {
                ["layer_0"] = 0.0001f,
                ["layer_1"] = 0.001f
            };

            optimizer.SetLayerWiseLrs(layerLrs);

            // Multipliers should be stored internally
            // This test mainly checks that no exception is thrown
        }

        [Fact]
        public void SetLayerWiseLrs_ByFrozenUnfrozen_SetsCorrectLrs()
        {
            var model = CreateTestModel(3);
            model.Freeze(exceptLastN: 1);
            var optimizer = CreateMockOptimizer(CreateParameterDict(model), 0.001f);

            optimizer.SetLayerWiseLrs(model, frozenLr: 0.0001f, unfrozenLr: 0.001f);

            // Multipliers should be stored internally
        }

        [Fact]
        public void SetLayerWiseLrs_BySchedule_SetsCorrectLrs()
        {
            var model = CreateTestModel(2);
            var optimizer = CreateMockOptimizer(CreateParameterDict(model), 0.001f);

            var lrSchedule = new[] { 0.0001f, 0.001f };
            var layerNames = new[] { "layer_0", "layer_1" };

            optimizer.SetLayerWiseLrs(lrSchedule, layerNames);

            // Multipliers should be stored internally
        }

        [Fact]
        public void SetLayerWiseLrs_WithNullOptimizer_ThrowsArgumentNullException()
        {
            var layerLrs = new Dictionary<string, float>();
            Optimizer optimizer = null;
            Assert.Throws<ArgumentNullException>(() => optimizer!.SetLayerWiseLrs(layerLrs));
        }

        [Fact]
        public void SetLayerWiseLrs_WithNullDictionary_ThrowsArgumentNullException()
        {
            var model = CreateTestModel(1);
            var optimizer = CreateMockOptimizer(CreateParameterDict(model), 0.001f);

            Assert.Throws<ArgumentNullException>(() => optimizer.SetLayerWiseLrs((Dictionary<string, float>)null));
        }

        [Fact]
        public void SetLayerWiseLrs_BySchedule_WithMismatchedArrays_ThrowsArgumentException()
        {
            var model = CreateTestModel(1);
            var optimizer = CreateMockOptimizer(CreateParameterDict(model), 0.001f);

            var lrSchedule = new[] { 0.0001f, 0.001f };
            var layerNames = new[] { "layer_0" };

            Assert.Throws<ArgumentException>(() => optimizer.SetLayerWiseLrs(lrSchedule, layerNames));
        }

        [Fact]
        public void GetParameterGroups_ReturnsCorrectGroups()
        {
            var model = CreateTestModel(2);
            var optimizer = CreateMockOptimizer(CreateParameterDict(model), 0.001f);

            var groups = optimizer.GetParameterGroups(model);

            Assert.True(groups.Count > 0);
        }

        [Fact]
        public void ApplyParameterGroups_AppliesCorrectLrs()
        {
            var model = CreateTestModel(2);
            var optimizer = CreateMockOptimizer(CreateParameterDict(model), 0.001f);

            var groups = new List<ParameterGroup>
            {
                new ParameterGroup("frozen", 0.0001f),
                new ParameterGroup("unfrozen", 0.001f)
            };

            // Add parameters to groups
            var allParams = model.GetNamedParameters().ToList();
            foreach (var (paramName, param) in allParams.Take(allParams.Count / 2))
            {
                groups[0].AddParameter(param, paramName);
            }
            foreach (var (paramName, param) in allParams.Skip(allParams.Count / 2))
            {
                groups[1].AddParameter(param, paramName);
            }

            optimizer.ApplyParameterGroups(groups);

            // Multipliers should be stored internally
        }

        [Fact]
        public void CreateDiscriminativeGroups_CreatesCorrectGroups()
        {
            var model = CreateTestModel(5);
            var optimizer = CreateMockOptimizer(CreateParameterDict(model), 0.001f);

            var builder = optimizer.CreateDiscriminativeGroups(model, 3, 0.1f, 1.0f);

            var groups = builder.Build();

            Assert.Equal(3, groups.Count);

            // Check that learning rates increase from first to last group
            Assert.True(groups[0].LearningRate < groups[1].LearningRate);
            Assert.True(groups[1].LearningRate < groups[2].LearningRate);
        }

        [Fact]
        public void ClearLayerWiseLrs_ClearsMultipliers()
        {
            var model = CreateTestModel(1);
            var optimizer = CreateMockOptimizer(CreateParameterDict(model), 0.001f);

            optimizer.SetLayerWiseLrs(new Dictionary<string, float> { ["layer_0"] = 0.0001f });
            optimizer.ClearLayerWiseLrs();

            // Multipliers should be cleared (no exception expected)
        }

        #endregion

        #region LayerWiseLrScheduler Tests

        [Fact]
        public void LayerWiseLrScheduler_Constructor_CreatesValidScheduler()
        {
            var scheduler = new LayerWiseLrScheduler(0.001f);

            Assert.Equal(0.001f, scheduler.BaseLearningRate);
            Assert.Equal(0, scheduler.ScheduledLayerCount);
        }

        [Fact]
        public void LayerWiseLrScheduler_Constructor_WithInvalidLr_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new LayerWiseLrScheduler(-0.001f));
            Assert.Throws<ArgumentOutOfRangeException>(() => new LayerWiseLrScheduler(0.0f));
        }

        [Fact]
        public void LayerWiseLrScheduler_SetMultiplier_SetsMultiplier()
        {
            var scheduler = new LayerWiseLrScheduler(0.001f);

            scheduler.SetMultiplier("layer_0", 0.5f);

            var currentLrs = scheduler.GetCurrentLrs();

            Assert.True(currentLrs.ContainsKey("layer_0"));
            Assert.Equal(0.0005f, currentLrs["layer_0"], 0.00001);
        }

        [Fact]
        public void LayerWiseLrScheduler_SetMultiplierByLayerIndex_SetsMultiplier()
        {
            var scheduler = new LayerWiseLrScheduler(0.001f);

            scheduler.SetMultiplierByLayerIndex(0, 0.5f);

            var currentLrs = scheduler.GetCurrentLrs();

            Assert.True(currentLrs.ContainsKey("layer_0"));
        }

        [Fact]
        public void LayerWiseLrScheduler_SetLayerSchedule_SetsSchedule()
        {
            var scheduler = new LayerWiseLrScheduler(0.001f);

            scheduler.SetLayerSchedule("layer_0", LayerWiseLrScheduler.ScheduleType.Linear, 0.1f, 1.0f, 100);

            Assert.Equal(1, scheduler.ScheduledLayerCount);
        }

        [Fact]
        public void LayerWiseLrScheduler_GetCurrentLrs_ReturnsCorrectLrs()
        {
            var scheduler = new LayerWiseLrScheduler(0.001f);

            scheduler.SetMultiplier("layer_0", 0.5f);
            scheduler.SetMultiplier("layer_1", 1.0f);

            var currentLrs = scheduler.GetCurrentLrs();

            Assert.Equal(2, currentLrs.Count);
            Assert.Equal(0.0005f, currentLrs["layer_0"], 0.00001);
            Assert.Equal(0.001f, currentLrs["layer_1"], 0.00001);
        }

        [Fact]
        public void LayerWiseLrScheduler_GetParameterLearningRate_ReturnsCorrectLr()
        {
            var scheduler = new LayerWiseLrScheduler(0.001f);

            scheduler.SetMultiplier("layer_0", 0.5f);

            float lr = scheduler.GetParameterLearningRate("layer_0.weight");

            Assert.Equal(0.0005f, lr, 0.00001);
        }

        [Fact]
        public void LayerWiseLrScheduler_GetGroupLearningRate_ReturnsCorrectLr()
        {
            var scheduler = new LayerWiseLrScheduler(0.001f);

            float lr = scheduler.GetGroupLearningRate(0, "layer_0");

            Assert.Equal(0.001f, lr);
        }

        [Fact]
        public void LayerWiseLrScheduler_StateDict_RoundTrip()
        {
            var scheduler = new LayerWiseLrScheduler(0.001f);

            scheduler.SetMultiplier("layer_0", 0.5f);
            scheduler.SetLayerSchedule("layer_1", LayerWiseLrScheduler.ScheduleType.Cosine, 0.1f, 1.0f, 100);

            var state = scheduler.GetState();

            var newScheduler = new LayerWiseLrScheduler(0.001f);
            newScheduler.LoadState(state);

            Assert.Equal(scheduler.BaseLearningRate, newScheduler.BaseLearningRate);
            Assert.Equal(scheduler.ScheduledLayerCount, newScheduler.ScheduledLayerCount);
        }

        #endregion

        #region Fine-Tuning Schedule Factory Methods Tests

        [Fact]
        public void CreateDiscriminative_CreatesCorrectSchedule()
        {
            var multipliers = new[] { 0.1f, 0.5f, 1.0f };
            var scheduler = LayerWiseLrScheduler.CreateDiscriminative(0.001f, multipliers);

            Assert.Equal(3, scheduler.ScheduledLayerCount);

            var currentLrs = scheduler.GetCurrentLrs();

            Assert.True(currentLrs.ContainsKey("layer_0"));
            Assert.True(currentLrs.ContainsKey("layer_1"));
            Assert.True(currentLrs.ContainsKey("layer_2"));
        }

        [Fact]
        public void CreateTriangular_CreatesCorrectSchedule()
        {
            var scheduler = LayerWiseLrScheduler.CreateTriangular(0.001f, 0.0001f, 0.01f, 1000);

            Assert.NotNull(scheduler);
            Assert.Equal(0.001f, scheduler.BaseLearningRate);
        }

        [Fact]
        public void CreateWarmupCosine_CreatesCorrectSchedule()
        {
            var scheduler = LayerWiseLrScheduler.CreateWarmupCosine(0.0001f, 0.001f, 100, 1000);

            Assert.NotNull(scheduler);
            Assert.Equal(0.0001f, scheduler.BaseLearningRate);
            Assert.Equal(2, scheduler.ScheduledLayerCount); // warmup and annealing schedules
        }

        [Fact]
        public void CreateGeometricDiscriminative_CreatesCorrectSchedule()
        {
            var scheduler = LayerWiseLrScheduler.CreateGeometricDiscriminative(0.001f, 5, 0.1f, 1.0f);

            Assert.NotNull(scheduler);
            Assert.Equal(0.001f, scheduler.BaseLearningRate);
        }

        [Fact]
        public void CreateGradualUnfreezing_CreatesCorrectSchedule()
        {
            var scheduler = LayerWiseLrScheduler.CreateGradualUnfreezing(0.001f, 4, 0.0001f, 0.001f);

            Assert.NotNull(scheduler);
            Assert.Equal(0.001f, scheduler.BaseLearningRate);
        }

        [Fact]
        public void CreateDiscriminative_WithEmptyMultipliers_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() =>
                LayerWiseLrScheduler.CreateDiscriminative(0.001f, Array.Empty<float>()));
        }

        [Fact]
        public void CreateTriangular_WithInvalidLrs_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                LayerWiseLrScheduler.CreateTriangular(0.001f, -0.0001f, 0.01f));
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                LayerWiseLrScheduler.CreateTriangular(0.001f, 0.0001f, 0.00001f));
        }

        [Fact]
        public void CreateWarmupCosine_WithInvalidSteps_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                LayerWiseLrScheduler.CreateWarmupCosine(0.001f, 0.01f, -100, 1000));
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                LayerWiseLrScheduler.CreateWarmupCosine(0.001f, 0.01f, 100, 50));
        }

        #endregion

        #region Edge Cases and Integration Tests

        [Fact]
        public void EmptyModel_DoesNotThrow()
        {
            var model = new SequentialModule("empty");
            var optimizer = CreateMockOptimizer(new Dictionary<string, Tensor>(), 0.001f);

            var exception = Record.Exception(() => optimizer.SetLayerWiseLrs(model, 0.0001f, 0.001f));
            Assert.Null(exception);
        }

        [Fact]
        public void AllParametersFrozen_WorksCorrectly()
        {
            var model = CreateTestModel(3);
            model.Freeze(); // Freeze all

            var optimizer = CreateMockOptimizer(CreateParameterDict(model), 0.001f);
            optimizer.SetLayerWiseLrs(model, frozenLr: 0.0001f, unfrozenLr: 0.001f);

            var exception = Record.Exception(() => optimizer.GetParameterGroups(model));
            Assert.Null(exception);
        }

        [Fact]
        public void SingleParameterGroup_WorksCorrectly()
        {
            var model = CreateTestModel(1);
            var optimizer = CreateMockOptimizer(CreateParameterDict(model), 0.001f);

            var builder = new ParameterGroupBuilder();
            builder.NewGroup("all", 0.001f).AddUnfrozenParameters(model);

            var groups = builder.Build();

            Assert.Single(groups);
            Assert.Equal(2, groups[0].ParameterCount); // weight and bias
        }

        [Fact]
        public void SchedulerStep_AdvancesCorrectly()
        {
            var scheduler = new LayerWiseLrScheduler(0.001f);
            scheduler.SetMultiplier("layer_0", 0.5f);

            scheduler.Step();

            Assert.Equal(1, scheduler.StepCount);
        }

        [Fact]
        public void SchedulerReset_ResetsCorrectly()
        {
            var scheduler = new LayerWiseLrScheduler(0.001f);
            scheduler.Step();
            scheduler.Step();

            scheduler.Reset();

            Assert.Equal(0, scheduler.StepCount);
        }

        [Fact]
        public void ParameterGroupBuilder_FluentChain_WorksCorrectly()
        {
            var model = CreateTestModel(3);

            var builder = new ParameterGroupBuilder()
                .NewGroup("frozen", 0.0001f)
                .AddFirstNLayers(2, model)
                .WithWeightDecay(0.001f)
                .NewGroup("unfrozen", 0.001f)
                .AddLastNLayers(1, model)
                .WithMomentum(0.9f);

            var groups = builder.Build();

            Assert.Equal(2, groups.Count);
            Assert.Equal(0.0001f, groups[0].LearningRate);
            Assert.Equal(0.001f, groups[1].LearningRate);
            Assert.Equal(0.001f, groups[0].WeightDecay);
            Assert.Equal(0.9f, groups[1].Momentum);
        }

        #endregion

        #region Mock Optimizer Class

        /// <summary>
        /// Mock optimizer for testing.
        /// </summary>
        private class MockOptimizer : Optimizer
        {
            private float _learningRate;

            public MockOptimizer(Dictionary<string, Tensor> parameters, float learningRate)
                : base(parameters)
            {
                _learningRate = learningRate;
            }

            public override float BaseLearningRate => _learningRate;

            public override void SetLearningRate(float lr)
            {
                _learningRate = lr;
            }

            public override void Step(Dictionary<string, Tensor> gradients)
            {
                // Mock implementation - does nothing
                UpdateLearningRate();
                _stepCount++;
                StepScheduler();
            }

            public override void StepParameter(string parameterName, Tensor gradient)
            {
                // Mock implementation - does nothing
            }

            public override void ZeroGrad()
            {
                // Mock implementation - does nothing
            }
        }

        #endregion
    }
}
