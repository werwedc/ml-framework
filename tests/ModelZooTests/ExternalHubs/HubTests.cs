using MLFramework.ModelVersioning;
using MLFramework.ModelZoo.ExternalHubs;

namespace MLFramework.Tests.ModelZooTests.ExternalHubs;

/// <summary>
/// Unit tests for ModelIdParser.
/// </summary>
[TestClass]
public class ModelIdParserTests
{
    [TestMethod]
    public void Parse_LocalModel_NoHubPrefix()
    {
        // Arrange
        var modelId = "bert-base-uncased";

        // Act
        var result = ModelIdParser.Parse(modelId);

        // Assert
        Assert.IsNotNull(result);
        Assert.IsNull(result.HubName);
        Assert.AreEqual("bert-base-uncased", result.ModelName);
        Assert.IsNull(result.Version);
        Assert.IsNull(result.Variant);
        Assert.IsTrue(result.IsLocal);
    }

    [TestMethod]
    public void Parse_HuggingFaceModel_WithHubPrefix()
    {
        // Arrange
        var modelId = "hub:huggingface/bert-base-uncased";

        // Act
        var result = ModelIdParser.Parse(modelId);

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual("huggingface", result.HubName);
        Assert.AreEqual("bert-base-uncased", result.ModelName);
        Assert.IsNull(result.Version);
        Assert.IsNull(result.Variant);
        Assert.IsFalse(result.IsLocal);
    }

    [TestMethod]
    public void Parse_TensorFlowModel_WithVersion()
    {
        // Arrange
        var modelId = "hub:tensorflow/resnet50-v2/classification";

        // Act
        var result = ModelIdParser.Parse(modelId);

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual("tensorflow", result.HubName);
        Assert.AreEqual("resnet50", result.ModelName);
        Assert.AreEqual("v2", result.Version);
        Assert.AreEqual("classification", result.Variant);
        Assert.IsFalse(result.IsLocal);
    }

    [TestMethod]
    public void Parse_ONNXModel_WithVersion()
    {
        // Arrange
        var modelId = "hub:onnx/vgg16-7";

        // Act
        var result = ModelIdParser.Parse(modelId);

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual("onnx", result.HubName);
        Assert.AreEqual("vgg16", result.ModelName);
        Assert.AreEqual("7", result.Version);
        Assert.IsNull(result.Variant);
        Assert.IsFalse(result.IsLocal);
    }

    [TestMethod]
    public void Parse_CustomRegistryModel()
    {
        // Arrange
        var modelId = "hub:custom:my-registry/efficientnet";

        // Act
        var result = ModelIdParser.Parse(modelId);

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual("custom:my-registry", result.HubName);
        Assert.AreEqual("efficientnet", result.ModelName);
        Assert.IsNull(result.Version);
        Assert.IsNull(result.Variant);
        Assert.IsFalse(result.IsLocal);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Parse_NullModelId_ThrowsException()
    {
        // Act
        ModelIdParser.Parse(null!);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Parse_EmptyModelId_ThrowsException()
    {
        // Act
        ModelIdParser.Parse(string.Empty);
    }

    [TestMethod]
    [ExpectedException(typeof(FormatException))]
    public void Parse_InvalidFormat_ThrowsException()
    {
        // Arrange
        var modelId = "hub:huggingface";

        // Act
        ModelIdParser.Parse(modelId);
    }

    [TestMethod]
    public void TryParse_ValidModelId_ReturnsTrue()
    {
        // Arrange
        var modelId = "hub:huggingface/bert-base-uncased";

        // Act
        var success = ModelIdParser.TryParse(modelId, out var result);

        // Assert
        Assert.IsTrue(success);
        Assert.IsNotNull(result);
        Assert.AreEqual("huggingface", result.HubName);
    }

    [TestMethod]
    public void TryParse_InvalidModelId_ReturnsFalse()
    {
        // Arrange
        var modelId = "hub:huggingface";

        // Act
        var success = ModelIdParser.TryParse(modelId, out var result);

        // Assert
        Assert.IsFalse(success);
        Assert.IsNull(result);
    }

    [TestMethod]
    public void IsLocalModel_LocalModel_ReturnsTrue()
    {
        // Arrange
        var modelId = "bert-base-uncased";

        // Act
        var result = ModelIdParser.IsLocalModel(modelId);

        // Assert
        Assert.IsTrue(result);
    }

    [TestMethod]
    public void IsLocalModel_HubModel_ReturnsFalse()
    {
        // Arrange
        var modelId = "hub:huggingface/bert-base-uncased";

        // Act
        var result = ModelIdParser.IsLocalModel(modelId);

        // Assert
        Assert.IsFalse(result);
    }

    [TestMethod]
    public void FullId_LocalModel_ReturnsModelName()
    {
        // Arrange
        var modelId = "bert-base-uncased";
        var components = ModelIdParser.Parse(modelId);

        // Act
        var result = components.FullId;

        // Assert
        Assert.AreEqual("bert-base-uncased", result);
    }

    [TestMethod]
    public void FullId_HubModel_ReturnsHubPrefix()
    {
        // Arrange
        var modelId = "hub:huggingface/bert-base-uncased";
        var components = ModelIdParser.Parse(modelId);

        // Act
        var result = components.FullId;

        // Assert
        Assert.AreEqual("hub:huggingface/bert-base-uncased", result);
    }
}

/// <summary>
/// Unit tests for hub authentication classes.
/// </summary>
[TestClass]
public class HubAuthenticationTests
{
    [TestMethod]
    public void AnonymousAuth_Valid_ReturnsTrue()
    {
        // Arrange
        var auth = new AnonymousAuth();

        // Act
        var result = auth.IsValid();

        // Assert
        Assert.IsTrue(result);
    }

    [TestMethod]
    public void AnonymousAuth_GetAuthHeader_ReturnsEmpty()
    {
        // Arrange
        var auth = new AnonymousAuth();

        // Act
        var (headerName, headerValue) = auth.GetAuthHeader();

        // Assert
        Assert.AreEqual(string.Empty, headerName);
        Assert.AreEqual(string.Empty, headerValue);
    }

    [TestMethod]
    public void ApiKeyAuth_ValidKey_ReturnsTrue()
    {
        // Arrange
        var auth = new ApiKeyAuth("test-api-key");

        // Act
        var result = auth.IsValid();

        // Assert
        Assert.IsTrue(result);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void ApiKeyAuth_NullKey_ThrowsException()
    {
        // Act
        new ApiKeyAuth(null!);
    }

    [TestMethod]
    public void ApiKeyAuth_GetAuthHeader_ReturnsBearerToken()
    {
        // Arrange
        var auth = new ApiKeyAuth("test-api-key");

        // Act
        var (headerName, headerValue) = auth.GetAuthHeader();

        // Assert
        Assert.AreEqual("Authorization", headerName);
        Assert.AreEqual("Bearer test-api-key", headerValue);
    }

    [TestMethod]
    public void TokenAuth_ValidToken_ReturnsTrue()
    {
        // Arrange
        var auth = new TokenAuth("test-token");

        // Act
        var result = auth.IsValid();

        // Assert
        Assert.IsTrue(result);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TokenAuth_NullToken_ThrowsException()
    {
        // Act
        new TokenAuth(null!);
    }

    [TestMethod]
    public void TokenAuth_GetAuthHeader_ReturnsBearerToken()
    {
        // Arrange
        var auth = new TokenAuth("test-token", "Bearer");

        // Act
        var (headerName, headerValue) = auth.GetAuthHeader();

        // Assert
        Assert.AreEqual("Authorization", headerName);
        Assert.AreEqual("Bearer test-token", headerValue);
    }

    [TestMethod]
    public void TokenAuth_CustomTokenType_ReturnsCorrectHeader()
    {
        // Arrange
        var auth = new TokenAuth("test-token", "Token");

        // Act
        var (headerName, headerValue) = auth.GetAuthHeader();

        // Assert
        Assert.AreEqual("Authorization", headerName);
        Assert.AreEqual("Token test-token", headerValue);
    }
}

/// <summary>
/// Unit tests for hub configuration classes.
/// </summary>
[TestClass]
public class HubConfigurationTests
{
    [TestMethod]
    public void HubConfiguration_DefaultValues_AreValid()
    {
        // Arrange
        var config = new HubConfiguration();

        // Act
        var isValid = config.IsValid();

        // Assert
        Assert.IsTrue(isValid);
        Assert.AreEqual(300, config.TimeoutSeconds);
        Assert.AreEqual(3, config.MaxRetries);
        Assert.AreEqual(5, config.RetryDelaySeconds);
        Assert.IsTrue(config.VerifySsl);
    }

    [TestMethod]
    public void HuggingFaceHubConfiguration_DefaultValues_AreValid()
    {
        // Arrange
        var config = new HuggingFaceHubConfiguration();

        // Act
        var isValid = config.IsValid();

        // Assert
        Assert.IsTrue(isValid);
        Assert.AreEqual("https://huggingface.co", config.BaseUrl);
        Assert.AreEqual("models", config.DefaultRepository);
        Assert.AreEqual(config.BaseUrl, config.EffectiveBaseUrl);
    }

    [TestMethod]
    public void HuggingFaceHubConfiguration_WithMirror_ReturnsMirrorUrl()
    {
        // Arrange
        var config = new HuggingFaceHubConfiguration
        {
            BaseUrl = "https://huggingface.co",
            MirrorUrl = "https://mirror.huggingface.co",
            UseMirror = true
        };

        // Act
        var effectiveUrl = config.EffectiveBaseUrl;

        // Assert
        Assert.AreEqual("https://mirror.huggingface.co", effectiveUrl);
    }

    [TestMethod]
    public void TensorFlowHubConfiguration_DefaultValues_AreValid()
    {
        // Arrange
        var config = new TensorFlowHubConfiguration();

        // Act
        var isValid = config.IsValid();

        // Assert
        Assert.IsTrue(isValid);
        Assert.AreEqual("https://tfhub.dev", config.BaseUrl);
        Assert.IsTrue(config.UseCompression);
        Assert.AreEqual("gzip", config.CompressionFormat);
        Assert.AreEqual("saved_model", config.PreferredFormat);
    }

    [TestMethod]
    public void ONNXHubConfiguration_DefaultValues_AreValid()
    {
        // Arrange
        var config = new ONNXHubConfiguration();

        // Act
        var isValid = config.IsValid();

        // Assert
        Assert.IsTrue(isValid);
        Assert.AreEqual("https://github.com/onnx/models", config.BaseUrl);
        Assert.AreEqual("latest", config.DefaultVersion);
    }

    [TestMethod]
    public void HubConfiguration_InvalidTimeout_ReturnsFalse()
    {
        // Arrange
        var config = new HubConfiguration
        {
            TimeoutSeconds = -1
        };

        // Act
        var isValid = config.IsValid();

        // Assert
        Assert.IsFalse(isValid);
    }
}

/// <summary>
/// Unit tests for HubRegistry.
/// </summary>
[TestClass]
public class HubRegistryTests
{
    private HubRegistry _registry = null!;

    [TestInitialize]
    public void Setup()
    {
        _registry = new HubRegistry();
    }

    [TestMethod]
    public void RegisterHub_ValidHub_AddsToRegistry()
    {
        // Arrange
        var hub = new MockHub("test-hub");

        // Act
        _registry.RegisterHub(hub);

        // Assert
        Assert.AreEqual(1, _registry.Count);
        Assert.IsTrue(_registry.IsHubRegistered("test-hub"));
        Assert.IsNotNull(_registry.GetHub("test-hub"));
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void RegisterHub_NullHub_ThrowsException()
    {
        // Act
        _registry.RegisterHub(null!);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void RegisterHub_DuplicateHub_ThrowsException()
    {
        // Arrange
        var hub1 = new MockHub("test-hub");
        var hub2 = new MockHub("test-hub");

        // Act
        _registry.RegisterHub(hub1);
        _registry.RegisterHub(hub2);
    }

    [TestMethod]
    public void UnregisterHub_ExistingHub_RemovesFromRegistry()
    {
        // Arrange
        var hub = new MockHub("test-hub");
        _registry.RegisterHub(hub);

        // Act
        var result = _registry.UnregisterHub("test-hub");

        // Assert
        Assert.IsTrue(result);
        Assert.AreEqual(0, _registry.Count);
        Assert.IsFalse(_registry.IsHubRegistered("test-hub"));
    }

    [TestMethod]
    public void UnregisterHub_NonExistentHub_ReturnsFalse()
    {
        // Act
        var result = _registry.UnregisterHub("non-existent");

        // Assert
        Assert.IsFalse(result);
    }

    [TestMethod]
    public void GetHub_ExistingHub_ReturnsHub()
    {
        // Arrange
        var hub = new MockHub("test-hub");
        _registry.RegisterHub(hub);

        // Act
        var result = _registry.GetHub("test-hub");

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual("test-hub", result.HubName);
    }

    [TestMethod]
    public void GetHub_NonExistentHub_ReturnsNull()
    {
        // Act
        var result = _registry.GetHub("non-existent");

        // Assert
        Assert.IsNull(result);
    }

    [TestMethod]
    public void GetHubForModel_CanHandle_ReturnsCorrectHub()
    {
        // Arrange
        var hub1 = new MockHub("hub1") { CanHandleModelIds = new[] { "model1" } };
        var hub2 = new MockHub("hub2") { CanHandleModelIds = new[] { "model2" } };
        _registry.RegisterHub(hub1);
        _registry.RegisterHub(hub2);

        // Act
        var result = _registry.GetHubForModel("model1");

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual("hub1", result.HubName);
    }

    [TestMethod]
    public void ListHubs_EmptyRegistry_ReturnsEmptyArray()
    {
        // Act
        var result = _registry.ListHubs();

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual(0, result.Length);
    }

    [TestMethod]
    public void ListHubs_WithHubs_ReturnsAllHubNames()
    {
        // Arrange
        _registry.RegisterHub(new MockHub("hub1"));
        _registry.RegisterHub(new MockHub("hub2"));

        // Act
        var result = _registry.ListHubs();

        // Assert
        Assert.AreEqual(2, result.Length);
        CollectionAssert.Contains(result, "hub1");
        CollectionAssert.Contains(result, "hub2");
    }

    [TestMethod]
    public void GetDefaultHub_NoHubs_ReturnsNull()
    {
        // Act
        var result = _registry.GetDefaultHub();

        // Assert
        Assert.IsNull(result);
    }

    [TestMethod]
    public void GetDefaultHub_FirstHubBecomesDefault()
    {
        // Arrange
        var hub = new MockHub("test-hub");
        _registry.RegisterHub(hub);

        // Act
        var result = _registry.GetDefaultHub();

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual("test-hub", result.HubName);
    }

    [TestMethod]
    public void SetDefaultHub_ExistingHub_Succeeds()
    {
        // Arrange
        _registry.RegisterHub(new MockHub("hub1"));
        _registry.RegisterHub(new MockHub("hub2"));

        // Act
        var result = _registry.SetDefaultHub("hub2");

        // Assert
        Assert.IsTrue(result);
        Assert.AreEqual("hub2", _registry.GetDefaultHub()?.HubName);
    }

    [TestMethod]
    public void SetDefaultHub_NonExistentHub_ReturnsFalse()
    {
        // Arrange
        _registry.RegisterHub(new MockHub("hub1"));

        // Act
        var result = _registry.SetDefaultHub("non-existent");

        // Assert
        Assert.IsFalse(result);
    }

    [TestMethod]
    public void Clear_RemovesAllHubs()
    {
        // Arrange
        _registry.RegisterHub(new MockHub("hub1"));
        _registry.RegisterHub(new MockHub("hub2"));

        // Act
        _registry.Clear();

        // Assert
        Assert.AreEqual(0, _registry.Count);
        Assert.IsNull(_registry.GetDefaultHub());
    }

    [TestMethod]
    public void RegisterDefaultHubs_RegistersAllDefaultHubs()
    {
        // Act
        _registry.RegisterDefaultHubs();

        // Assert
        Assert.IsTrue(_registry.IsHubRegistered("huggingface"));
        Assert.IsTrue(_registry.IsHubRegistered("tensorflow"));
        Assert.IsTrue(_registry.IsHubRegistered("onnx"));
    }
}

/// <summary>
/// Unit tests for HuggingFaceHub.
/// </summary>
[TestClass]
public class HuggingFaceHubTests
{
    [TestMethod]
    public void Constructor_DefaultConfiguration_CreatesValidInstance()
    {
        // Act
        var hub = new HuggingFaceHub();

        // Assert
        Assert.AreEqual("huggingface", hub.HubName);
        Assert.IsNotNull(hub.Configuration);
        Assert.IsNotNull(hub.Authentication);
    }

    [TestMethod]
    public void Constructor_WithConfiguration_UsesProvidedConfig()
    {
        // Arrange
        var config = new HuggingFaceHubConfiguration { BaseUrl = "https://custom.hf.co" };

        // Act
        var hub = new HuggingFaceHub(config);

        // Assert
        Assert.AreEqual("https://custom.hf.co", ((HuggingFaceHubConfiguration)hub.Configuration).BaseUrl);
    }

    [TestMethod]
    public void CanHandle_HuggingFaceModelId_ReturnsTrue()
    {
        // Arrange
        var hub = new HuggingFaceHub();

        // Act
        var result = hub.CanHandle("hub:huggingface/bert-base-uncased");

        // Assert
        Assert.IsTrue(result);
    }

    [TestMethod]
    public void CanHandle_HFAlias_ReturnsTrue()
    {
        // Arrange
        var hub = new HuggingFaceHub();

        // Act
        var result = hub.CanHandle("hub:hf/bert-base-uncased");

        // Assert
        Assert.IsTrue(result);
    }

    [TestMethod]
    public void CanHandle_OtherHubModelId_ReturnsFalse()
    {
        // Arrange
        var hub = new HuggingFaceHub();

        // Act
        var result = hub.CanHandle("hub:tensorflow/resnet50");

        // Assert
        Assert.IsFalse(result);
    }
}

/// <summary>
/// Unit tests for TensorFlowHub.
/// </summary>
[TestClass]
public class TensorFlowHubTests
{
    [TestMethod]
    public void Constructor_DefaultConfiguration_CreatesValidInstance()
    {
        // Act
        var hub = new TensorFlowHub();

        // Assert
        Assert.AreEqual("tensorflow", hub.HubName);
        Assert.IsNotNull(hub.Configuration);
        Assert.IsNotNull(hub.Authentication);
    }

    [TestMethod]
    public void CanHandle_TensorFlowModelId_ReturnsTrue()
    {
        // Arrange
        var hub = new TensorFlowHub();

        // Act
        var result = hub.CanHandle("hub:tensorflow/resnet50");

        // Assert
        Assert.IsTrue(result);
    }

    [TestMethod]
    public void CanHandle_TFAlias_ReturnsTrue()
    {
        // Arrange
        var hub = new TensorFlowHub();

        // Act
        var result = hub.CanHandle("hub:tf/resnet50");

        // Assert
        Assert.IsTrue(result);
    }
}

/// <summary>
/// Unit tests for ONNXHub.
/// </summary>
[TestClass]
public class ONNXHubTests
{
    [TestMethod]
    public void Constructor_DefaultConfiguration_CreatesValidInstance()
    {
        // Act
        var hub = new ONNXHub();

        // Assert
        Assert.AreEqual("onnx", hub.HubName);
        Assert.IsNotNull(hub.Configuration);
        Assert.IsNotNull(hub.Authentication);
    }

    [TestMethod]
    public void CanHandle_ONNXModelId_ReturnsTrue()
    {
        // Arrange
        var hub = new ONNXHub();

        // Act
        var result = hub.CanHandle("hub:onnx/vgg16-7");

        // Assert
        Assert.IsTrue(result);
    }

    [TestMethod]
    public void CanHandle_OtherHubModelId_ReturnsFalse()
    {
        // Arrange
        var hub = new ONNXHub();

        // Act
        var result = hub.CanHandle("hub:huggingface/bert");

        // Assert
        Assert.IsFalse(result);
    }
}

/// <summary>
/// Unit tests for ModelZooExternalExtensions.
/// </summary>
[TestClass]
public class ModelZooExternalExtensionsTests
{
    private ModelZoo _modelZoo = null!;

    [TestInitialize]
    public void Setup()
    {
        _modelZoo = new ModelZoo();
    }

    [TestMethod]
    public void HubRegistry_ReturnsSingleton()
    {
        // Act
        var registry1 = ModelZooExternalExtensions.HubRegistry;
        var registry2 = ModelZooExternalExtensions.HubRegistry;

        // Assert
        Assert.AreSame(registry1, registry2);
    }

    [TestMethod]
    public void InitializeExternalHubs_RegistersDefaultHubs()
    {
        // Act
        ModelZooExternalExtensions.InitializeExternalHubs();

        // Assert
        Assert.IsTrue(ModelZooExternalExtensions.HubRegistry.IsHubRegistered("huggingface"));
        Assert.IsTrue(ModelZooExternalExtensions.HubRegistry.IsHubRegistered("tensorflow"));
        Assert.IsTrue(ModelZooExternalExtensions.HubRegistry.IsHubRegistered("onnx"));
    }

    [TestMethod]
    public void RegisterHub_ValidHub_RegistersInRegistry()
    {
        // Arrange
        var hub = new MockHub("test-hub");

        // Act
        _modelZoo.RegisterHub(hub);

        // Assert
        Assert.IsTrue(ModelZooExternalExtensions.HubRegistry.IsHubRegistered("test-hub"));
    }

    [TestMethod]
    public void ListHubs_WithRegisteredHubs_ReturnsHubNames()
    {
        // Arrange
        ModelZooExternalExtensions.HubRegistry.RegisterHub(new MockHub("hub1"));
        ModelZooExternalExtensions.HubRegistry.RegisterHub(new MockHub("hub2"));

        // Act
        var result = _modelZoo.ListHubs();

        // Assert
        Assert.IsTrue(result.Contains("hub1"));
        Assert.IsTrue(result.Contains("hub2"));
    }

    [TestMethod]
    public void GetHub_ExistingHub_ReturnsHub()
    {
        // Arrange
        ModelZooExternalExtensions.HubRegistry.RegisterHub(new MockHub("test-hub"));

        // Act
        var result = _modelZoo.GetHub("test-hub");

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual("test-hub", result.HubName);
    }

    [TestMethod]
    [ExpectedException(typeof(InvalidOperationException))]
    public async Task GetModelMetadataFromHubAsync_InvalidModelId_ThrowsException()
    {
        // Act
        await _modelZoo.GetModelMetadataFromHubAsync("hub:nonexistent/model");
    }
}

#region Mock Classes for Testing

/// <summary>
/// Mock implementation of IModelHub for testing.
/// </summary>
internal class MockHub : IModelHub
{
    public MockHub(string hubName)
    {
        HubName = hubName;
    }

    public string HubName { get; }

    public IHubAuthentication? Authentication => new AnonymousAuth();

    public HubConfiguration Configuration => new HubConfiguration();

    public string[]? CanHandleModelIds { get; set; }

    public Task<ModelMetadata> GetModelMetadataAsync(string modelId)
    {
        return Task.FromResult(new ModelMetadata { ModelName = modelId });
    }

    public Task<Stream> DownloadModelAsync(string modelId, IProgress<double>? progress = null)
    {
        return Task.FromResult<Stream>(new MemoryStream());
    }

    public Task<bool> ModelExistsAsync(string modelId)
    {
        return Task.FromResult(true);
    }

    public Task<string[]> ListModelsAsync(string? filter = null)
    {
        return Task.FromResult(new[] { "model1", "model2" });
    }

    public bool CanHandle(string modelId)
    {
        if (CanHandleModelIds != null)
        {
            return CanHandleModelIds.Contains(modelId);
        }
        return modelId.Contains(HubName, StringComparison.OrdinalIgnoreCase);
    }
}

#endregion
