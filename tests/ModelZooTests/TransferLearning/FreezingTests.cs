using System;
using System.Linq;
using MLFramework.ModelZoo.TransferLearning;
using MLFramework.NN;
using Xunit;

namespace MLFramework.Tests.ModelZooTests.TransferLearning
{
    /// <summary>
    /// Unit tests for layer freezing and unfreezing functionality.
    /// </summary>
    public class FreezingTests
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
        /// Creates a model with hierarchical structure for testing.
        /// </summary>
        private SequentialModule CreateHierarchicalModel()
        {
            var model = new SequentialModule("hierarchical_model");

            var branch1 = new SequentialModule("branch1");
            branch1.Add(CreateMockModule("branch1_layer1"));
            branch1.Add(CreateMockModule("branch1_layer2"));

            var branch2 = new SequentialModule("branch2");
            branch2.Add(CreateMockModule("branch2_layer1"));
            branch2.Add(CreateMockModule("branch2_layer2"));

            model.Add(CreateMockModule("stem"));
            model.Add(branch1);
            model.Add(branch2);
            model.Add(CreateMockModule("head"));

            return model;
        }

        #endregion

        #region Basic Freeze/Unfreeze Tests

        [Fact]
        public void Freeze_DisablesGradientComputationForAllParameters()
        {
            var model = CreateTestModel(3);

            // Initially all parameters should have gradient tracking enabled
            Assert.All(model.GetParameters(), p => Assert.True(p.RequiresGrad));

            // Freeze the model
            model.Freeze();

            // All parameters should now have gradient tracking disabled
            Assert.All(model.GetParameters(), p => Assert.False(p.RequiresGrad));
        }

        [Fact]
        public void Unfreeze_EnablesGradientComputationForAllParameters()
        {
            var model = CreateTestModel(3);
            model.Freeze();

            // All parameters should be frozen
            Assert.All(model.GetParameters(), p => Assert.False(p.RequiresGrad));

            // Unfreeze the model
            model.Unfreeze();

            // All parameters should now have gradient tracking enabled
            Assert.All(model.GetParameters(), p => Assert.True(p.RequiresGrad));
        }

        [Fact]
        public void Freeze_WithNullModule_ThrowsArgumentNullException()
        {
            Module? module = null;
            Assert.Throws<ArgumentNullException>(() => module!.Freeze());
        }

        [Fact]
        public void Unfreeze_WithNullModule_ThrowsArgumentNullException()
        {
            Module? module = null;
            Assert.Throws<ArgumentNullException>(() => module!.Unfreeze());
        }

        #endregion

        #region Freeze Except Last N Tests

        [Fact]
        public void Freeze_WithExceptLastN_FreezesAllExceptLastN()
        {
            var model = CreateTestModel(5);
            int exceptLastN = 2;

            model.Freeze(exceptLastN);

            var allLayers = model.GetAllModules().ToList();
            var frozenLayers = model.GetFrozenLayers().ToList();
            var unfrozenLayers = model.GetUnfrozenLayers().ToList();

            // Should have frozen layers equal to total - exceptLastN
            Assert.Equal(5 - exceptLastN, frozenLayers.Count);
            Assert.Equal(exceptLastN, unfrozenLayers.Count);

            // Last N layers should be unfrozen
            for (int i = allLayers.Count - exceptLastN; i < allLayers.Count; i++)
            {
                Assert.Contains(allLayers[i], unfrozenLayers);
            }

            // First layers should be frozen
            for (int i = 0; i < allLayers.Count - exceptLastN; i++)
            {
                Assert.Contains(allLayers[i], frozenLayers);
            }
        }

        [Fact]
        public void Freeze_WithExceptLastN_Zero_FreezesAll()
        {
            var model = CreateTestModel(3);
            model.Freeze(0);

            var frozenLayers = model.GetFrozenLayers().ToList();
            var unfrozenLayers = model.GetUnfrozenLayers().ToList();

            Assert.Equal(3, frozenLayers.Count);
            Assert.Equal(0, unfrozenLayers.Count);
        }

        [Fact]
        public void Freeze_WithExceptLastN_GreaterThanTotal_UnfreezesAll()
        {
            var model = CreateTestModel(3);
            model.Freeze(10);

            var frozenLayers = model.GetFrozenLayers().ToList();
            var unfrozenLayers = model.GetUnfrozenLayers().ToList();

            Assert.Equal(0, frozenLayers.Count);
            Assert.Equal(3, unfrozenLayers.Count);
        }

        [Fact]
        public void Freeze_WithNegativeExceptLastN_ThrowsArgumentOutOfRangeException()
        {
            var model = CreateTestModel(3);
            Assert.Throws<ArgumentOutOfRangeException>(() => model.Freeze(-1));
        }

        #endregion

        #region Freeze/Unfreeze By Name Tests

        [Fact]
        public void FreezeByName_FreezesSpecificLayers()
        {
            var model = CreateTestModel(5);
            model.FreezeByName("layer_0", "layer_2", "layer_4");

            var frozenLayers = model.GetFrozenLayers().ToList();
            var unfrozenLayers = model.GetUnfrozenLayers().ToList();

            Assert.Equal(3, frozenLayers.Count);
            Assert.Equal(2, unfrozenLayers.Count);

            Assert.Contains("layer_0", frozenLayers.Select(l => l.Name));
            Assert.Contains("layer_2", frozenLayers.Select(l => l.Name));
            Assert.Contains("layer_4", frozenLayers.Select(l => l.Name));
            Assert.Contains("layer_1", unfrozenLayers.Select(l => l.Name));
            Assert.Contains("layer_3", unfrozenLayers.Select(l => l.Name));
        }

        [Fact]
        public void UnfreezeByName_UnfreezesSpecificLayers()
        {
            var model = CreateTestModel(5);
            model.Freeze(); // Freeze all first
            model.UnfreezeByName("layer_1", "layer_3");

            var frozenLayers = model.GetFrozenLayers().ToList();
            var unfrozenLayers = model.GetUnfrozenLayers().ToList();

            Assert.Equal(3, frozenLayers.Count);
            Assert.Equal(2, unfrozenLayers.Count);

            Assert.Contains("layer_1", unfrozenLayers.Select(l => l.Name));
            Assert.Contains("layer_3", unfrozenLayers.Select(l => l.Name));
            Assert.Contains("layer_0", frozenLayers.Select(l => l.Name));
        }

        [Fact]
        public void FreezeByName_WithNonExistentLayer_DoesNotThrow()
        {
            var model = CreateTestModel(3);
            var exception = Record.Exception(() => model.FreezeByName("non_existent_layer"));
            Assert.Null(exception);
        }

        [Fact]
        public void FreezeByName_WithNullNames_ThrowsArgumentNullException()
        {
            var model = CreateTestModel(3);
            Assert.Throws<ArgumentNullException>(() => model.FreezeByName(null!));
        }

        #endregion

        #region Freeze/Unfreeze By Pattern Tests

        [Fact]
        public void FreezeByNamePattern_FreezesMatchingLayers()
        {
            var model = CreateTestModel(5);
            model.FreezeByNamePattern("^layer_[0-2]$");

            var frozenLayers = model.GetFrozenLayers().ToList();
            var unfrozenLayers = model.GetUnfrozenLayers().ToList();

            Assert.Equal(3, frozenLayers.Count);
            Assert.Equal(2, unfrozenLayers.Count);

            Assert.Contains("layer_0", frozenLayers.Select(l => l.Name));
            Assert.Contains("layer_1", frozenLayers.Select(l => l.Name));
            Assert.Contains("layer_2", frozenLayers.Select(l => l.Name));
            Assert.Contains("layer_3", unfrozenLayers.Select(l => l.Name));
            Assert.Contains("layer_4", unfrozenLayers.Select(l => l.Name));
        }

        [Fact]
        public void FreezeByNamePattern_WithWildcard_MatchesMultipleLayers()
        {
            var model = CreateTestModel(5);
            model.FreezeByNamePattern("layer_[135]"); // Matches layers 1, 3, 5 (but 5 doesn't exist)

            var frozenLayers = model.GetFrozenLayers().ToList();
            var unfrozenLayers = model.GetUnfrozenLayers().ToList();

            Assert.Equal(2, frozenLayers.Count);
            Assert.Contains("layer_1", frozenLayers.Select(l => l.Name));
            Assert.Contains("layer_3", frozenLayers.Select(l => l.Name));
        }

        [Fact]
        public void FreezeByNamePattern_WithInvalidPattern_ThrowsArgumentException()
        {
            var model = CreateTestModel(3);
            Assert.Throws<ArgumentException>(() => model.FreezeByNamePattern("[invalid("));
        }

        [Fact]
        public void FreezeByNamePattern_WithNullPattern_ThrowsArgumentNullException()
        {
            var model = CreateTestModel(3);
            Assert.Throws<ArgumentException>(() => model.FreezeByNamePattern(null!));
        }

        [Fact]
        public void UnfreezeByNamePattern_UnfreezesMatchingLayers()
        {
            var model = CreateTestModel(5);
            model.Freeze(); // Freeze all first
            model.UnfreezeByNamePattern("^layer_[0-2]$");

            var frozenLayers = model.GetFrozenLayers().ToList();
            var unfrozenLayers = model.GetUnfrozenLayers().ToList();

            Assert.Equal(2, frozenLayers.Count);
            Assert.Equal(3, unfrozenLayers.Count);
            Assert.Contains("layer_0", unfrozenLayers.Select(l => l.Name));
            Assert.Contains("layer_1", unfrozenLayers.Select(l => l.Name));
            Assert.Contains("layer_2", unfrozenLayers.Select(l => l.Name));
        }

        #endregion

        #region Query Frozen State Tests

        [Fact]
        public void GetFrozenLayers_ReturnsAllFrozenLayers()
        {
            var model = CreateTestModel(5);
            model.Freeze(exceptLastN: 2);

            var frozenLayers = model.GetFrozenLayers().ToList();
            Assert.Equal(3, frozenLayers.Count);
            Assert.DoesNotContain("layer_3", frozenLayers.Select(l => l.Name));
            Assert.DoesNotContain("layer_4", frozenLayers.Select(l => l.Name));
        }

        [Fact]
        public void GetUnfrozenLayers_ReturnsAllUnfrozenLayers()
        {
            var model = CreateTestModel(5);
            model.Freeze(exceptLastN: 2);

            var unfrozenLayers = model.GetUnfrozenLayers().ToList();
            Assert.Equal(2, unfrozenLayers.Count);
            Assert.Contains("layer_3", unfrozenLayers.Select(l => l.Name));
            Assert.Contains("layer_4", unfrozenLayers.Select(l => l.Name));
        }

        [Fact]
        public void GetFrozenStateSummary_ReturnsCorrectSummary()
        {
            var model = CreateTestModel(5);
            model.Freeze(exceptLastN: 2);

            var summary = model.GetFrozenStateSummary();

            Assert.Equal(5, summary.TotalLayers);
            Assert.Equal(3, summary.FrozenLayers);
            Assert.Equal(2, summary.UnfrozenLayers);
            Assert.Equal(3 * 2, summary.FrozenParameters); // 2 params per layer (weight and bias)
            Assert.Equal(2 * 2, summary.UnfrozenParameters);
            Assert.Equal(60.0, summary.FrozenParameterPercentage, 0.1);
            Assert.Equal(40.0, summary.UnfrozenParameterPercentage, 0.1);
        }

        [Fact]
        public void PrintFrozenState_PrintsToConsole()
        {
            var model = CreateTestModel(3);
            model.Freeze(exceptLastN: 1);

            // This should not throw an exception
            var exception = Record.Exception(() => model.PrintFrozenState());
            Assert.Null(exception);
        }

        #endregion

        #region Layer Selection Helper Tests

        [Fact]
        public void LayerSelectionHelper_SelectByName_ReturnsCorrectLayers()
        {
            var model = CreateTestModel(5);
            var selected = LayerSelectionHelper.SelectByName(model, "layer_1", "layer_3").ToList();

            Assert.Equal(2, selected.Count);
            Assert.Contains(selected, l => l.Name == "layer_1");
            Assert.Contains(selected, l => l.Name == "layer_3");
        }

        [Fact]
        public void LayerSelectionHelper_SelectByPattern_ReturnsMatchingLayers()
        {
            var model = CreateTestModel(5);
            var selected = LayerSelectionHelper.SelectByPattern(model, "^layer_[0-2]$").ToList();

            Assert.Equal(3, selected.Count);
        }

        [Fact]
        public void LayerSelectionHelper_SelectByIndex_ReturnsCorrectLayer()
        {
            var model = CreateTestModel(5);
            var layer = LayerSelectionHelper.SelectByIndex(model, 2);

            Assert.Equal("layer_2", layer.Name);
        }

        [Fact]
        public void LayerSelectionHelper_SelectByIndex_OutOfRange_ThrowsArgumentOutOfRangeException()
        {
            var model = CreateTestModel(3);
            Assert.Throws<ArgumentOutOfRangeException>(() => LayerSelectionHelper.SelectByIndex(model, 10));
        }

        [Fact]
        public void LayerSelectionHelper_SelectByRange_ReturnsCorrectLayers()
        {
            var model = CreateTestModel(5);
            var selected = LayerSelectionHelper.SelectByRange(model, 1, 3).ToList();

            Assert.Equal(3, selected.Count);
            Assert.Equal("layer_1", selected[0].Name);
            Assert.Equal("layer_2", selected[1].Name);
            Assert.Equal("layer_3", selected[2].Name);
        }

        [Fact]
        public void LayerSelectionHelper_SelectByType_ReturnsCorrectType()
        {
            var model = CreateTestModel(5);
            var selected = LayerSelectionHelper.SelectByType<LinearHead>(model).ToList();

            Assert.Equal(5, selected.Count);
            Assert.All(selected, l => Assert.IsType<LinearHead>(l));
        }

        [Fact]
        public void LayerSelectionHelper_SelectLastN_ReturnsLastNLayers()
        {
            var model = CreateTestModel(5);
            var selected = LayerSelectionHelper.SelectLastN(model, 2).ToList();

            Assert.Equal(2, selected.Count);
            Assert.Equal("layer_3", selected[0].Name);
            Assert.Equal("layer_4", selected[1].Name);
        }

        [Fact]
        public void LayerSelectionHelper_SelectFirstN_ReturnsFirstNLayers()
        {
            var model = CreateTestModel(5);
            var selected = LayerSelectionHelper.SelectFirstN(model, 2).ToList();

            Assert.Equal(2, selected.Count);
            Assert.Equal("layer_0", selected[0].Name);
            Assert.Equal("layer_1", selected[1].Name);
        }

        #endregion

        #region Gradual Unfreezing Scheduler Tests

        [Fact]
        public void GradualUnfreezingScheduler_Initialize_FreezesAllLayers()
        {
            var model = CreateTestModel(5);
            var thresholds = new[] { 0.25, 0.5, 0.75, 1.0 };
            var scheduler = new GradualUnfreezingScheduler(model, thresholds);

            scheduler.Initialize();

            var frozenLayers = model.GetFrozenLayers().ToList();
            Assert.Equal(5, frozenLayers.Count);
        }

        [Fact]
        public void GradualUnfreezingScheduler_UpdateUnfreezing_UnfreezesProgressively()
        {
            var model = CreateTestModel(5);
            var thresholds = new[] { 0.25, 0.5, 0.75, 1.0 };
            var scheduler = new GradualUnfreezingScheduler(model, thresholds);

            scheduler.Initialize();

            // At 10% progress, no unfreezing yet
            bool advanced = scheduler.UpdateUnfreezing(0.1);
            Assert.False(advanced);
            Assert.Equal(0, model.GetUnfrozenLayers().Count());

            // At 25% progress, first stage (last 1 layer unfrozen)
            advanced = scheduler.UpdateUnfreezing(0.25);
            Assert.True(advanced);
            Assert.Equal(1, model.GetUnfrozenLayers().Count());

            // At 50% progress, second stage (last 2 layers unfrozen)
            advanced = scheduler.UpdateUnfreezing(0.5);
            Assert.True(advanced);
            Assert.Equal(2, model.GetUnfrozenLayers().Count());

            // At 75% progress, third stage (last 3 layers unfrozen)
            advanced = scheduler.UpdateUnfreezing(0.75);
            Assert.True(advanced);
            Assert.Equal(3, model.GetUnfrozenLayers().Count());

            // At 100% progress, fourth stage (last 4 layers unfrozen)
            advanced = scheduler.UpdateUnfreezing(1.0);
            Assert.True(advanced);
            Assert.Equal(4, model.GetUnfrozenLayers().Count());
        }

        [Fact]
        public void GradualUnfreezingScheduler_AdvanceToNextStage_UnfreezesNextLayer()
        {
            var model = CreateTestModel(5);
            var thresholds = new[] { 0.25, 0.5, 0.75, 1.0 };
            var scheduler = new GradualUnfreezingScheduler(model, thresholds);

            scheduler.Initialize();

            scheduler.AdvanceToNextStage();
            Assert.Equal(1, model.GetUnfrozenLayers().Count());

            scheduler.AdvanceToNextStage();
            Assert.Equal(2, model.GetUnfrozenLayers().Count());
        }

        [Fact]
        public void GradualUnfreezingScheduler_SetStage_SetsCorrectStage()
        {
            var model = CreateTestModel(5);
            var thresholds = new[] { 0.25, 0.5, 0.75, 1.0 };
            var scheduler = new GradualUnfreezingScheduler(model, thresholds);

            scheduler.Initialize();

            scheduler.SetStage(2);
            Assert.Equal(3, model.GetUnfrozenLayers().Count()); // Stage 2 unfreezes last 3 layers

            scheduler.SetStage(0);
            Assert.Equal(1, model.GetUnfrozenLayers().Count()); // Stage 0 unfreezes last 1 layer
        }

        [Fact]
        public void GradualUnfreezingScheduler_Reset_FreezesAllLayers()
        {
            var model = CreateTestModel(5);
            var thresholds = new[] { 0.25, 0.5, 0.75, 1.0 };
            var scheduler = new GradualUnfreezingScheduler(model, thresholds);

            scheduler.Initialize();
            scheduler.SetStage(2);

            scheduler.Reset();

            var frozenLayers = model.GetFrozenLayers().ToList();
            Assert.Equal(5, frozenLayers.Count);
            Assert.Equal(-1, scheduler.CurrentStage);
        }

        [Fact]
        public void GradualUnfreezingScheduler_CreateEvenlyDistributed_CreatesEvenThresholds()
        {
            var model = CreateTestModel(5);
            var scheduler = GradualUnfreezingScheduler.CreateEvenlyDistributed(model, 4);

            Assert.Equal(4, scheduler.TotalStages);
            Assert.Equal(0.25, scheduler.UnfreezeThresholds[0], 0.001);
            Assert.Equal(0.5, scheduler.UnfreezeThresholds[1], 0.001);
            Assert.Equal(0.75, scheduler.UnfreezeThresholds[2], 0.001);
            Assert.Equal(1.0, scheduler.UnfreezeThresholds[3], 0.001);
        }

        [Fact]
        public void GradualUnfreezingScheduler_OnUnfreezeStage_FiresEvent()
        {
            var model = CreateTestModel(5);
            var thresholds = new[] { 0.25, 0.5, 0.75, 1.0 };
            var scheduler = new GradualUnfreezingScheduler(model, thresholds);

            int eventCount = 0;
            int lastStage = -1;
            scheduler.OnUnfreezeStage += (stage, total) =>
            {
                eventCount++;
                lastStage = stage;
            };

            scheduler.Initialize();
            scheduler.UpdateUnfreezing(0.25);

            Assert.Equal(1, eventCount);
            Assert.Equal(0, lastStage);
        }

        [Fact]
        public void GradualUnfreezingScheduler_WithInvalidThresholds_ThrowsArgumentException()
        {
            var model = CreateTestModel(5);

            // Thresholds out of range
            Assert.Throws<ArgumentException>(() => new GradualUnfreezingScheduler(model, new[] { -0.1 }));
            Assert.Throws<ArgumentException>(() => new GradualUnfreezingScheduler(model, new[] { 1.5 }));

            // Unsorted thresholds
            Assert.Throws<ArgumentException>(() => new GradualUnfreezingScheduler(model, new[] { 0.5, 0.25 }));

            // Empty thresholds
            Assert.Throws<ArgumentException>(() => new GradualUnfreezingScheduler(model, Array.Empty<double>()));
        }

        [Fact]
        public void GradualUnfreezingScheduler_GetStateSummary_ReturnsCorrectSummary()
        {
            var model = CreateTestModel(5);
            var thresholds = new[] { 0.25, 0.5, 0.75, 1.0 };
            var scheduler = new GradualUnfreezingScheduler(model, thresholds);

            scheduler.Initialize();
            scheduler.SetStage(1);

            var summary = scheduler.GetStateSummary();
            Assert.Contains("Stage 2/4", summary);
            Assert.Contains("Unfrozen: 2/5", summary);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Freeze_EmptyModel_DoesNotThrow()
        {
            var model = new SequentialModule("empty_model");
            var exception = Record.Exception(() => model.Freeze());
            Assert.Null(exception);
        }

        [Fact]
        public void Freeze_SingleLayerModel_WorksCorrectly()
        {
            var model = CreateTestModel(1);
            model.Freeze();

            Assert.Equal(1, model.GetFrozenLayers().Count());
            Assert.Equal(0, model.GetUnfrozenLayers().Count());
        }

        [Fact]
        public void Freeze_HierarchicalModel_WorksCorrectly()
        {
            var model = CreateHierarchicalModel();
            model.Freeze(exceptLastN: 1);

            var frozenLayers = model.GetFrozenLayers().ToList();
            var unfrozenLayers = model.GetUnfrozenLayers().ToList();

            // Should have 1 unfrozen layer (the head)
            Assert.Equal(1, unfrozenLayers.Count);
            Assert.Contains("head", unfrozenLayers.Select(l => l.Name));
        }

        [Fact]
        public void FrozenStateSummary_ToString_ReturnsFormattedString()
        {
            var model = CreateTestModel(4);
            model.Freeze(exceptLastN: 1);

            var summary = model.GetFrozenStateSummary();
            var summaryStr = summary.ToString();

            Assert.Contains("Frozen: 6/8", summaryStr);
            Assert.Contains("Unfrozen: 2/8", summaryStr);
            Assert.Contains("Layers: 3 frozen, 1 unfrozen", summaryStr);
        }

        #endregion
    }
}
