using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using MLFramework.ModelZoo.Plugins;
using MLFramework.ModelZoo.Plugins.BuiltIn;
using Xunit;

namespace ModelZooTests.Plugins
{
    /// <summary>
    /// Comprehensive unit tests for the Model Zoo plugin architecture.
    /// </summary>
    public class PluginTests
    {
        #region Plugin Manager Tests

        [Fact]
        public void PluginManager_RegisterPlugin_Success()
        {
            // Arrange
            var manager = new PluginManager();
            var plugin = new ExampleRegistry();

            // Act
            manager.RegisterPlugin(plugin);

            // Assert
            Assert.Equal(1, manager.PluginCount);
            Assert.Contains(plugin.RegistryName, manager.ListPlugins());
        }

        [Fact]
        public void PluginManager_RegisterPlugin_ThrowsOnNull()
        {
            // Arrange
            var manager = new PluginManager();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => manager.RegisterPlugin(null));
        }

        [Fact]
        public void PluginManager_RegisterPlugin_ThrowsOnDuplicate()
        {
            // Arrange
            var manager = new PluginManager();
            var plugin1 = new ExampleRegistry();
            var plugin2 = new ExampleRegistry();

            // Act
            manager.RegisterPlugin(plugin1);

            // Assert
            Assert.Throws<InvalidOperationException>(() => manager.RegisterPlugin(plugin2));
        }

        [Fact]
        public void PluginManager_UnregisterPlugin_Success()
        {
            // Arrange
            var manager = new PluginManager();
            var plugin = new ExampleRegistry();
            manager.RegisterPlugin(plugin);

            // Act
            var result = manager.UnregisterPlugin(plugin.RegistryName);

            // Assert
            Assert.True(result);
            Assert.Equal(0, manager.PluginCount);
            Assert.DoesNotContain(plugin.RegistryName, manager.ListPlugins());
        }

        [Fact]
        public void PluginManager_UnregisterPlugin_ReturnsFalseOnNotFound()
        {
            // Arrange
            var manager = new PluginManager();

            // Act
            var result = manager.UnregisterPlugin("nonexistent");

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void PluginManager_GetPlugin_Success()
        {
            // Arrange
            var manager = new PluginManager();
            var plugin = new ExampleRegistry();
            manager.RegisterPlugin(plugin);

            // Act
            var retrieved = manager.GetPlugin(plugin.RegistryName);

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal(plugin.RegistryName, retrieved.RegistryName);
        }

        [Fact]
        public void PluginManager_GetPlugin_ReturnsNullOnNotFound()
        {
            // Arrange
            var manager = new PluginManager();

            // Act
            var retrieved = manager.GetPlugin("nonexistent");

            // Assert
            Assert.Null(retrieved);
        }

        [Fact]
        public void PluginManager_ListPlugins_Success()
        {
            // Arrange
            var manager = new PluginManager();
            var plugin1 = new ExampleRegistry();
            var plugin2 = new BuiltIn.MemoryRegistry();

            manager.RegisterPlugin(plugin1);
            manager.RegisterPlugin(plugin2);

            // Act
            var plugins = manager.ListPlugins();

            // Assert
            Assert.Equal(2, plugins.Count);
            Assert.Contains(plugin1.RegistryName, plugins);
            Assert.Contains(plugin2.RegistryName, plugins);
        }

        [Fact]
        public void PluginManager_FindPlugin_RespectsPriority()
        {
            // Arrange
            var manager = new PluginManager();
            var lowPriority = new ExampleRegistry(); // Priority 10
            var highPriority = new BuiltIn.MemoryRegistry(); // Priority 200

            manager.RegisterPlugin(lowPriority);
            manager.RegisterPlugin(highPriority);

            // Act - MemoryRegistry should be found first due to higher priority
            var plugin = manager.FindPlugin("example:transformer");

            // Assert
            Assert.NotNull(plugin);
            Assert.Equal(highPriority.RegistryName, plugin.RegistryName);
        }

        [Fact]
        public async Task PluginManager_GetAllModelMetadata_Success()
        {
            // Arrange
            var manager = new PluginManager();
            var registry = new ExampleRegistry();
            manager.RegisterPlugin(registry);

            // Act
            var metadata = await manager.GetAllModelMetadata("simple-nn");

            // Assert
            Assert.NotNull(metadata);
            Assert.Equal("simple-nn", metadata.ModelName);
        }

        [Fact]
        public async Task PluginManager_GetBestSource_Success()
        {
            // Arrange
            var manager = new PluginManager();
            var registry = new ExampleRegistry();
            manager.RegisterPlugin(registry);

            // Act
            var best = await manager.GetBestSource("simple-nn");

            // Assert
            Assert.NotNull(best);
            Assert.Equal(registry.RegistryName, best.RegistryName);
        }

        #endregion

        #region Extension Tests

        [Fact]
        public void PluginManager_RegisterExtension_Success()
        {
            // Arrange
            var manager = new PluginManager();
            var extension = new TestExtension();

            // Act
            manager.RegisterExtension(extension);

            // Assert
            Assert.Equal(1, manager.ExtensionCount);
        }

        [Fact]
        public void PluginManager_ExecuteExtensions_Success()
        {
            // Arrange
            var manager = new PluginManager();
            var extension = new TestExtension();
            manager.RegisterExtension(extension);

            var metadata = new MLFramework.ModelVersioning.ModelMetadata
            {
                ModelName = "test-model",
                Framework = "Test",
                Architecture = "TestArch"
            };

            // Act
            manager.ExecutePreDownloadExtensions(metadata).Wait();
            manager.ExecutePostDownloadExtensions(metadata, new MemoryStream()).Wait();
            manager.ExecutePreLoadExtensions(metadata).Wait();
            manager.ExecutePostLoadExtensions(metadata).Wait();

            // Assert - Extension should have been called
            Assert.True(extension.PreDownloadCalled);
            Assert.True(extension.PostDownloadCalled);
            Assert.True(extension.PreLoadCalled);
            Assert.True(extension.PostLoadCalled);
        }

        #endregion

        #region Memory Registry Tests

        [Fact]
        public async Task MemoryRegistry_AddModel_Success()
        {
            // Arrange
            var registry = new BuiltIn.MemoryRegistry();
            var metadata = new MLFramework.ModelVersioning.ModelMetadata
            {
                ModelName = "test-model",
                Framework = "Test",
                Architecture = "TestArch"
            };
            var data = new byte[] { 0x01, 0x02, 0x03 };

            // Act
            registry.AddModel("test-model", metadata, data);

            // Assert
            Assert.Equal(1, registry.Count);
            Assert.True(await registry.ModelExistsAsync("test-model"));
        }

        [Fact]
        public async Task MemoryRegistry_RemoveModel_Success()
        {
            // Arrange
            var registry = new BuiltIn.MemoryRegistry();
            var metadata = new MLFramework.ModelVersioning.ModelMetadata
            {
                ModelName = "test-model",
                Framework = "Test",
                Architecture = "TestArch"
            };
            var data = new byte[] { 0x01, 0x02, 0x03 };

            registry.AddModel("test-model", metadata, data);

            // Act
            var removed = registry.RemoveModel("test-model");

            // Assert
            Assert.True(removed);
            Assert.Equal(0, registry.Count);
            Assert.False(await registry.ModelExistsAsync("test-model"));
        }

        [Fact]
        public async Task MemoryRegistry_DownloadModel_Success()
        {
            // Arrange
            var registry = new BuiltIn.MemoryRegistry();
            var metadata = new MLFramework.ModelVersioning.ModelMetadata
            {
                ModelName = "test-model",
                Framework = "Test",
                Architecture = "TestArch"
            };
            var data = new byte[] { 0x01, 0x02, 0x03 };

            registry.AddModel("test-model", metadata, data);

            // Act
            var stream = await registry.DownloadModelAsync(metadata);
            var downloaded = new byte[data.Length];
            await stream.ReadAsync(downloaded, 0, data.Length);

            // Assert
            Assert.Equal(data, downloaded);
        }

        #endregion

        #region Authentication Tests

        [Fact]
        public void ApiKeyAuthentication_AuthenticatesRequest()
        {
            // Arrange
            var auth = new ApiKeyAuthentication("secret-key", "X-Api-Key");
            var request = new HttpRequestMessage(HttpMethod.Get, "http://example.com");

            // Act
            auth.Authenticate(request);

            // Assert
            Assert.True(request.Headers.Contains("X-Api-Key"));
            Assert.Equal("secret-key", request.Headers.GetValues("X-Api-Key").First());
        }

        [Fact]
        public void TokenAuthentication_AuthenticatesRequest()
        {
            // Arrange
            var auth = new TokenAuthentication("my-token", "Bearer");
            var request = new HttpRequestMessage(HttpMethod.Get, "http://example.com");

            // Act
            auth.Authenticate(request);

            // Assert
            Assert.NotNull(request.Headers.Authorization);
            Assert.Equal("Bearer", request.Headers.Authorization.Scheme);
            Assert.Equal("my-token", request.Headers.Authorization.Parameter);
        }

        [Fact]
        public void BasicAuthentication_AuthenticatesRequest()
        {
            // Arrange
            var auth = new BasicAuthentication("username", "password");
            var request = new HttpRequestMessage(HttpMethod.Get, "http://example.com");

            // Act
            auth.Authenticate(request);

            // Assert
            Assert.NotNull(request.Headers.Authorization);
            Assert.Equal("Basic", request.Headers.Authorization.Scheme);
            Assert.NotNull(request.Headers.Authorization.Parameter);
        }

        [Fact]
        public void CustomAuthentication_AuthenticatesRequest()
        {
            // Arrange
            bool authCalled = false;
            var auth = new CustomAuthentication(req =>
            {
                authCalled = true;
                req.Headers.Add("X-Custom", "custom-value");
            }, "Custom");

            var request = new HttpRequestMessage(HttpMethod.Get, "http://example.com");

            // Act
            auth.Authenticate(request);

            // Assert
            Assert.True(authCalled);
            Assert.True(request.Headers.Contains("X-Custom"));
        }

        #endregion

        #region Example Registry Tests

        [Fact]
        public async Task ExampleRegistry_ListModels_Success()
        {
            // Arrange
            var registry = new ExampleRegistry();

            // Act
            var models = await registry.ListModelsAsync();

            // Assert
            Assert.NotEmpty(models);
            Assert.Contains("simple-nn", models);
        }

        [Fact]
        public async Task ExampleRegistry_GetModelMetadata_Success()
        {
            // Arrange
            var registry = new ExampleRegistry();

            // Act
            var metadata = await registry.GetModelMetadataAsync("simple-nn");

            // Assert
            Assert.NotNull(metadata);
            Assert.Equal("simple-nn", metadata.ModelName);
        }

        [Fact]
        public async Task ExampleRegistry_DownloadModel_Success()
        {
            // Arrange
            var registry = new ExampleRegistry();
            var metadata = await registry.GetModelMetadataAsync("simple-nn");

            // Act
            var stream = await registry.DownloadModelAsync(metadata);

            // Assert
            Assert.NotNull(stream);
            Assert.True(stream.Length > 0);
        }

        [Fact]
        public async Task ExampleRegistry_ModelExists_Success()
        {
            // Arrange
            var registry = new ExampleRegistry();

            // Act
            var exists = await registry.ModelExistsAsync("simple-nn");

            // Assert
            Assert.True(exists);
        }

        [Fact]
        public async Task ExampleRegistry_CanHandle_Success()
        {
            // Arrange
            var registry = new ExampleRegistry();

            // Act & Assert
            Assert.True(registry.CanHandle("example:simple-nn"));
            Assert.True(registry.CanHandle("simple-nn"));
            Assert.False(registry.CanHandle("nonexistent-model"));
        }

        #endregion

        #region Plugin Configuration Tests

        [Fact]
        public void PluginConfigurationBase_LoadFromEnvironment_Success()
        {
            // Arrange
            Environment.SetEnvironmentVariable("TEST_PLUGIN_RegistryUrl", "http://test.com");
            Environment.SetEnvironmentVariable("TEST_PLUGIN_EnableCaching", "true");
            Environment.SetEnvironmentVariable("TEST_PLUGIN_CacheExpirationHours", "48");

            // Act
            var config = PluginConfigurationLoader.LoadFromEnvironment<PluginConfigurationBase>("TEST_PLUGIN");

            // Assert
            Assert.Equal("http://test.com", config.RegistryUrl);
            Assert.True(config.EnableCaching);
            Assert.Equal(48, config.CacheExpirationHours);

            // Cleanup
            Environment.SetEnvironmentVariable("TEST_PLUGIN_RegistryUrl", null);
            Environment.SetEnvironmentVariable("TEST_PLUGIN_EnableCaching", null);
            Environment.SetEnvironmentVariable("TEST_PLUGIN_CacheExpirationHours", null);
        }

        #endregion

        #region Test Helpers

        /// <summary>
        /// Test extension for unit testing extension lifecycle.
        /// </summary>
        private class TestExtension : ModelZooExtensionBase
        {
            public override string ExtensionName => "TestExtension";
            public override int Priority => 100;

            public bool PreDownloadCalled { get; private set; }
            public bool PostDownloadCalled { get; private set; }
            public bool PreLoadCalled { get; private set; }
            public bool PostLoadCalled { get; private set; }

            public override Task PreDownloadAsync(MLFramework.ModelVersioning.ModelMetadata metadata)
            {
                PreDownloadCalled = true;
                return Task.CompletedTask;
            }

            public override Task<Stream> PostDownloadAsync(MLFramework.ModelVersioning.ModelMetadata metadata, Stream stream)
            {
                PostDownloadCalled = true;
                return Task.FromResult<Stream>(null);
            }

            public override Task PreLoadAsync(MLFramework.ModelVersioning.ModelMetadata metadata)
            {
                PreLoadCalled = true;
                return Task.CompletedTask;
            }

            public override Task PostLoadAsync(MLFramework.ModelVersioning.ModelMetadata metadata)
            {
                PostLoadCalled = true;
                return Task.CompletedTask;
            }
        }

        #endregion
    }
}
