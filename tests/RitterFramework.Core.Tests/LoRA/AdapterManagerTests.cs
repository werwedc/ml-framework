using RitterFramework.Core.LoRA;
using RitterFramework.Core.Tensor;
using Xunit;

namespace RitterFramework.Core.Tests.LoRA
{
    public class AdapterManagerTests
    {
        private MockModel _model;
        private AdapterManager _manager;
        private LoraAdapter _testAdapter;

        public AdapterManagerTests()
        {
            _model = new MockModel();
            _manager = new AdapterManager(_model);
            _testAdapter = CreateTestAdapter("test_adapter");
        }

        [Fact]
        public void Constructor_ValidModel_CreatesManager()
        {
            var model = new MockModel();
            var manager = new AdapterManager(model);

            Assert.NotNull(manager);
            Assert.Equal(model, manager.BaseModel);
        }

        [Fact]
        public void Constructor_NullModel_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => new AdapterManager(null));
        }

        [Fact]
        public void LoadAdapter_AddsToLoadedAdapters()
        {
            _manager.LoadAdapter(_testAdapter);

            Assert.Contains("test_adapter", _manager.ListAdapters());
        }

        [Fact]
        public void LoadAdapter_NullAdapter_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => _manager.LoadAdapter(null as LoraAdapter));
        }

        [Fact]
        public void LoadAdapter_AdapterWithoutName_ThrowsArgumentException()
        {
            var adapter = CreateTestAdapter("");
            Assert.Throws<ArgumentException>(() => _manager.LoadAdapter(adapter));
        }

        [Fact]
        public void SetActiveAdapter_SetsSingleActiveAdapter()
        {
            _manager.LoadAdapter(_testAdapter);
            _manager.SetActiveAdapter("test_adapter");

            Assert.Equal(new[] { "test_adapter" }, _manager.ListActiveAdapters());
        }

        [Fact]
        public void SetActiveAdapter_AdapterNotLoaded_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => _manager.SetActiveAdapter("nonexistent"));
        }

        [Fact]
        public void SetActiveAdapter_ReplacesExistingActiveAdapters()
        {
            var adapter1 = CreateTestAdapter("adapter1");
            var adapter2 = CreateTestAdapter("adapter2");

            _manager.LoadAdapter(adapter1);
            _manager.LoadAdapter(adapter2);
            _manager.SetActiveAdapter("adapter1");

            _manager.SetActiveAdapter("adapter2");

            Assert.Equal(new[] { "adapter2" }, _manager.ListActiveAdapters());
        }

        [Fact]
        public void SetActiveAdapter_WithMultiple_SetsAllActive()
        {
            var adapter1 = CreateTestAdapter("adapter1");
            var adapter2 = CreateTestAdapter("adapter2");

            _manager.LoadAdapter(adapter1);
            _manager.LoadAdapter(adapter2);

            _manager.SetActiveAdapter("adapter1", "adapter2");

            Assert.Equal(2, _manager.ListActiveAdapters().Count);
            Assert.Contains("adapter1", _manager.ListActiveAdapters());
            Assert.Contains("adapter2", _manager.ListActiveAdapters());
        }

        [Fact]
        public void ActivateAdapter_AddsToActiveSet()
        {
            var adapter1 = CreateTestAdapter("adapter1");
            var adapter2 = CreateTestAdapter("adapter2");

            _manager.LoadAdapter(adapter1);
            _manager.LoadAdapter(adapter2);

            _manager.SetActiveAdapter("adapter1");
            _manager.ActivateAdapter("adapter2");

            Assert.Contains("adapter2", _manager.ListActiveAdapters());
            Assert.Contains("adapter1", _manager.ListActiveAdapters());
        }

        [Fact]
        public void ActivateAdapter_AdapterNotLoaded_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => _manager.ActivateAdapter("nonexistent"));
        }

        [Fact]
        public void DeactivateAdapter_RemovesFromActiveSet()
        {
            var adapter1 = CreateTestAdapter("adapter1");
            var adapter2 = CreateTestAdapter("adapter2");

            _manager.LoadAdapter(adapter1);
            _manager.LoadAdapter(adapter2);

            _manager.SetActiveAdapter("adapter1", "adapter2");
            _manager.DeactivateAdapter("adapter1");

            Assert.Equal(new[] { "adapter2" }, _manager.ListActiveAdapters());
        }

        [Fact]
        public void DeactivateAdapter_AdapterNotActive_DoesNotThrow()
        {
            var adapter = CreateTestAdapter("adapter");
            _manager.LoadAdapter(adapter);

            var exception = Record.Exception(() => _manager.DeactivateAdapter("adapter"));
            Assert.Null(exception);
        }

        [Fact]
        public void UnloadAdapter_RemovesFromLoadedAndActive()
        {
            _manager.LoadAdapter(_testAdapter);
            _manager.SetActiveAdapter("test_adapter");

            _manager.UnloadAdapter("test_adapter");

            Assert.DoesNotContain("test_adapter", _manager.ListAdapters());
            Assert.DoesNotContain("test_adapter", _manager.ListActiveAdapters());
        }

        [Fact]
        public void UnloadAdapter_AdapterNotLoaded_DoesNotThrow()
        {
            var exception = Record.Exception(() => _manager.UnloadAdapter("nonexistent"));
            Assert.Null(exception);
        }

        [Fact]
        public void GetAdapter_ReturnsLoadedAdapter()
        {
            _manager.LoadAdapter(_testAdapter);
            var adapter = _manager.GetAdapter("test_adapter");

            Assert.Equal(_testAdapter, adapter);
        }

        [Fact]
        public void GetAdapter_AdapterNotLoaded_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => _manager.GetAdapter("nonexistent"));
        }

        [Fact]
        public void ListAdapters_ReturnsAllLoadedAdapters()
        {
            var adapter1 = CreateTestAdapter("adapter1");
            var adapter2 = CreateTestAdapter("adapter2");
            var adapter3 = CreateTestAdapter("adapter3");

            _manager.LoadAdapter(adapter1);
            _manager.LoadAdapter(adapter2);
            _manager.LoadAdapter(adapter3);

            var adapters = _manager.ListAdapters();
            Assert.Equal(3, adapters.Count);
            Assert.Contains("adapter1", adapters);
            Assert.Contains("adapter2", adapters);
            Assert.Contains("adapter3", adapters);
        }

        [Fact]
        public void ListActiveAdapters_ReturnsOnlyActiveAdapters()
        {
            var adapter1 = CreateTestAdapter("adapter1");
            var adapter2 = CreateTestAdapter("adapter2");
            var adapter3 = CreateTestAdapter("adapter3");

            _manager.LoadAdapter(adapter1);
            _manager.LoadAdapter(adapter2);
            _manager.LoadAdapter(adapter3);

            _manager.SetActiveAdapter("adapter1");
            _manager.ActivateAdapter("adapter3");

            var active = _manager.ListActiveAdapters();
            Assert.Equal(2, active.Count);
            Assert.Contains("adapter1", active);
            Assert.DoesNotContain("adapter2", active);
            Assert.Contains("adapter3", active);
        }

        private LoraAdapter CreateTestAdapter(string name)
        {
            var config = new LoraConfig(rank: 8, alpha: 16, dropout: 0.0f, targetModules: new[] { "layer1" });
            var adapter = new LoraAdapter(name, config);

            // Add some test weights
            var loraA = Core.Tensor.Tensor.Zeros(new[] { 8, 64 });
            var loraB = Core.Tensor.Tensor.Zeros(new[] { 128, 8 });
            adapter.AddModuleWeights("layer1", new LoraModuleWeights(loraA, loraB));

            return adapter;
        }

        // Mock model class for testing
        private class MockModel : IModule
        {
            public string Name { get; set; } = "mock_model";
        }
    }
}
