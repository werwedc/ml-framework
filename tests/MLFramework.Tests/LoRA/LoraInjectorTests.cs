using MLFramework.LoRA;
using MLFramework.Modules;
using MLFramework.NN;
using RitterFramework.Core.LoRA;
using Xunit;

namespace MLFramework.Tests.LoRA
{
    /// <summary>
    /// Unit tests for LoraInjector class
    /// </summary>
    public class LoraInjectorTests
    {
        private class MockHierarchicalModule : IHierarchicalModule
        {
            public string Name { get; set; }
            public IHierarchicalModule Parent { get; set; }
            private readonly List<IHierarchicalModule> _children;

            public MockHierarchicalModule(string name = "mock_module")
            {
                Name = name;
                _children = new List<IHierarchicalModule>();
            }

            public IEnumerable<IHierarchicalModule> Children()
            {
                return _children;
            }

            public void AddChild(IHierarchicalModule child)
            {
                _children.Add(child);
                child.Parent = this;
            }

            public void ReplaceChild(string name, IHierarchicalModule newChild)
            {
                for (int i = 0; i < _children.Count; i++)
                {
                    if (_children[i].Name == name)
                    {
                        _children[i] = newChild;
                        newChild.Parent = this;
                        return;
                    }
                }
            }
        }

        [Fact]
        public void Inject_WithNullModel_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => LoraInjector.Inject((IHierarchicalModule)null!, new LoraConfig()));
        }

        [Fact]
        public void Inject_WithNullConfig_UsesDefaultConfig()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var linearWrapper = new LinearWrapper(new Linear(64, 128), "q_proj");
            model.AddChild(linearWrapper);

            // Act - should not throw, uses default config
            var exception = Record.Exception(() => LoraInjector.Inject(model, null));
            Assert.Null(exception);
        }

        [Fact]
        public void Inject_ReplacesTargetLinearLayers()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var config = new LoraConfig { Rank = 8, Alpha = 16, TargetModules = new[] { "q_proj" } };

            var linear1 = new Linear(64, 128);
            var linear2 = new Linear(128, 64);
            var wrapper1 = new LinearWrapper(linear1, "q_proj");
            var wrapper2 = new LinearWrapper(linear2, "k_proj");

            model.AddChild(wrapper1);
            model.AddChild(wrapper2);

            // Act
            LoraInjector.Inject(model, config);

            // Assert
            var children = model.Children().ToList();
            Assert.IsType<LinearWrapper>(children[0]);
            var newWrapper1 = (LinearWrapper)children[0];
            Assert.IsType<LoraLinear>(newWrapper1.Module);

            var newWrapper2 = (LinearWrapper)children[1];
            Assert.IsType<Linear>(newWrapper2.Module); // Should not be replaced
        }

        [Fact]
        public void Inject_WithWildcard_ReplacesMatchingLayers()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var config = new LoraConfig { Rank = 8, Alpha = 16, TargetModules = new[] { "*_proj" } };

            var wrapper1 = new LinearWrapper(new Linear(64, 128), "q_proj");
            var wrapper2 = new LinearWrapper(new Linear(128, 64), "k_proj");
            var wrapper3 = new LinearWrapper(new Linear(64, 64), "not_matching");

            model.AddChild(wrapper1);
            model.AddChild(wrapper2);
            model.AddChild(wrapper3);

            // Act
            LoraInjector.Inject(model, config);

            // Assert
            var children = model.Children().ToList();
            Assert.IsType<LoraLinear>(((LinearWrapper)children[0]).Module);
            Assert.IsType<LoraLinear>(((LinearWrapper)children[1]).Module);
            Assert.IsType<Linear>(((LinearWrapper)children[2]).Module);
        }

        [Fact]
        public void Inject_WithMultipleTargets_ReplacesAllMatchingLayers()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var config = new LoraConfig { Rank = 8, Alpha = 16, TargetModules = new[] { "q_proj", "v_proj" } };

            var wrapper1 = new LinearWrapper(new Linear(64, 128), "q_proj");
            var wrapper2 = new LinearWrapper(new Linear(128, 64), "k_proj");
            var wrapper3 = new LinearWrapper(new Linear(64, 128), "v_proj");

            model.AddChild(wrapper1);
            model.AddChild(wrapper2);
            model.AddChild(wrapper3);

            // Act
            LoraInjector.Inject(model, config);

            // Assert
            var children = model.Children().ToList();
            Assert.IsType<LoraLinear>(((LinearWrapper)children[0]).Module);
            Assert.IsType<Linear>(((LinearWrapper)children[1]).Module);
            Assert.IsType<LoraLinear>(((LinearWrapper)children[2]).Module);
        }

        [Fact]
        public void Inject_AlreadyHasLoRA_SkipsAlreadyInjectedLayers()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var config = new LoraConfig { Rank = 8, Alpha = 16, TargetModules = new[] { "q_proj" } };

            var linear = new Linear(64, 128);
            var loraLinear = new LoraLinear(linear, 8, 16);
            var wrapper = new LinearWrapper(loraLinear, "q_proj");

            model.AddChild(wrapper);

            // Act - should not throw
            var exception = Record.Exception(() => LoraInjector.Inject(model, config));
            Assert.Null(exception);

            // Assert
            var children = model.Children().ToList();
            var resultWrapper = (LinearWrapper)children[0];
            Assert.Same(loraLinear, resultWrapper.Module); // Same instance, not replaced
        }

        [Fact]
        public void Inject_WithHierarchicalModel_RecursivelyProcessesChildren()
        {
            // Arrange
            var root = new MockHierarchicalModule("root");
            var child = new MockHierarchicalModule("child");
            var grandChild = new MockHierarchicalModule("grandchild");

            var wrapper = new LinearWrapper(new Linear(64, 128), "q_proj");

            root.AddChild(child);
            child.AddChild(grandChild);
            grandChild.AddChild(wrapper);

            var config = new LoraConfig { Rank = 8, Alpha = 16, TargetModules = new[] { "q_proj" } };

            // Act
            LoraInjector.Inject(root, config);

            // Assert
            var grandChildren = grandChild.Children().ToList();
            Assert.IsType<LoraLinear>(((LinearWrapper)grandChildren[0]).Module);
        }

        [Fact]
        public void Remove_WithNullModel_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => LoraInjector.Remove((IHierarchicalModule)null!));
        }

        [Fact]
        public void Remove_RestoresOriginalLayers()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var config = new LoraConfig { Rank = 8, Alpha = 16, TargetModules = new[] { "q_proj" } };

            var originalLinear = new Linear(64, 128);
            var wrapper = new LinearWrapper(originalLinear, "q_proj");
            model.AddChild(wrapper);

            LoraInjector.Inject(model, config);

            // Act
            LoraInjector.Remove(model);

            // Assert
            var children = model.Children().ToList();
            var resultWrapper = (LinearWrapper)children[0];
            Assert.IsType<Linear>(resultWrapper.Module);
            Assert.Same(originalLinear, resultWrapper.Module);
        }

        [Fact]
        public void Remove_ModelWithoutLoRA_DoesNotThrow()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var wrapper = new LinearWrapper(new Linear(64, 128), "q_proj");
            model.AddChild(wrapper);

            // Act & Assert
            var exception = Record.Exception(() => LoraInjector.Remove(model));
            Assert.Null(exception);
        }

        [Fact]
        public void Remove_WithMultipleLoRALayers_RestoresAll()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var config = new LoraConfig { Rank = 8, Alpha = 16, TargetModules = new[] { "q_proj", "v_proj" } };

            var wrapper1 = new LinearWrapper(new Linear(64, 128), "q_proj");
            var wrapper2 = new LinearWrapper(new Linear(128, 64), "v_proj");

            model.AddChild(wrapper1);
            model.AddChild(wrapper2);

            LoraInjector.Inject(model, config);

            // Act
            LoraInjector.Remove(model);

            // Assert
            var children = model.Children().ToList();
            Assert.IsType<Linear>(((LinearWrapper)children[0]).Module);
            Assert.IsType<Linear>(((LinearWrapper)children[1]).Module);
        }

        [Fact]
        public void HasLoRA_WithNullModel_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => LoraInjector.HasLoRA((IHierarchicalModule)null!));
        }

        [Fact]
        public void HasLoRA_ReturnsTrueAfterInjection()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var config = new LoraConfig();
            var wrapper = new LinearWrapper(new Linear(64, 128), "q_proj");
            model.AddChild(wrapper);

            // Act & Assert
            Assert.False(LoraInjector.HasLoRA(model));

            LoraInjector.Inject(model, config);
            Assert.True(LoraInjector.HasLoRA(model));
        }

        [Fact]
        public void HasLoRA_ReturnsFalseForModelWithoutLoRA()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var wrapper = new LinearWrapper(new Linear(64, 128), "q_proj");
            model.AddChild(wrapper);

            // Act & Assert
            Assert.False(LoraInjector.HasLoRA(model));
        }

        [Fact]
        public void GetLoRALayers_WithNullModel_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => LoraInjector.GetLoRALayers((IHierarchicalModule)null!));
        }

        [Fact]
        public void GetLoRALayers_ReturnsAllLoRALayers()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var config = new LoraConfig { Rank = 8, Alpha = 16, TargetModules = new[] { "q_proj", "v_proj" } };

            var wrapper1 = new LinearWrapper(new Linear(64, 128), "q_proj");
            var wrapper2 = new LinearWrapper(new Linear(128, 64), "v_proj");
            var wrapper3 = new LinearWrapper(new Linear(64, 64), "k_proj");

            model.AddChild(wrapper1);
            model.AddChild(wrapper2);
            model.AddChild(wrapper3);

            LoraInjector.Inject(model, config);

            // Act
            var loraLayers = LoraInjector.GetLoRALayers(model);

            // Assert
            Assert.Equal(2, loraLayers.Count);
            Assert.All(loraLayers, layer => Assert.IsType<LoraLinear>(layer));
        }

        [Fact]
        public void GetLoRALayers_ReturnsEmptyListForModelWithoutLoRA()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var wrapper = new LinearWrapper(new Linear(64, 128), "q_proj");
            model.AddChild(wrapper);

            // Act
            var loraLayers = LoraInjector.GetLoRALayers(model);

            // Assert
            Assert.Empty(loraLayers);
        }

        [Fact]
        public void Inject_WithEmptyTargetModules_UsesDefaultTargets()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var config = new LoraConfig { Rank = 8, Alpha = 16, TargetModules = new string[0] };

            var wrapper1 = new LinearWrapper(new Linear(64, 128), "q_proj");
            var wrapper2 = new LinearWrapper(new Linear(128, 64), "v_proj");

            model.AddChild(wrapper1);
            model.AddChild(wrapper2);

            // Act - should use default targets (q_proj and v_proj)
            LoraInjector.Inject(model, config);

            // Assert
            var children = model.Children().ToList();
            Assert.IsType<LoraLinear>(((LinearWrapper)children[0]).Module);
            Assert.IsType<LoraLinear>(((LinearWrapper)children[1]).Module);
        }

        [Fact]
        public void Inject_ValidatesConfigBeforeInjection()
        {
            // Arrange
            var model = new MockHierarchicalModule();
            var config = new LoraConfig { Rank = -1 }; // Invalid rank
            var wrapper = new LinearWrapper(new Linear(64, 128), "q_proj");
            model.AddChild(wrapper);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => LoraInjector.Inject(model, config));
        }
    }
}
