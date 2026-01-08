using MLFramework.ModelVersioning;
using MLFramework.Serving.Deployment;
using MLFramework.Serving.Routing;
using Moq;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Test fixture that sets up all services for integration testing.
    /// </summary>
    public class IntegrationTestFixture : IDisposable
    {
        public IModelRegistry Registry { get; }
        public IModelVersionManager VersionManager { get; }
        public IVersionRouter Router { get; }
        public IModelHotSwapper HotSwapper { get; }
        public Mock<IVersionRouter> MockRouter { get; }
        public Mock<IModelVersionManager> MockVersionManager { get; }
        public Mock<IModelRegistry> MockRegistry { get; }

        public IntegrationTestFixture()
        {
            // Create mocks
            MockVersionManager = new Mock<IModelVersionManager>();
            MockRouter = new Mock<IVersionRouter>();
            MockRegistry = new Mock<IModelRegistry>();

            // Setup default behaviors
            SetupDefaultBehaviors();

            // Create services with mocks
            VersionManager = MockVersionManager.Object;
            Router = MockRouter.Object;
            Registry = MockRegistry.Object;
            HotSwapper = new ModelHotSwapper(VersionManager, Router);

            // Initialize with test data
            InitializeTestData();
        }

        private void SetupDefaultBehaviors()
        {
            // Setup version manager defaults
            MockVersionManager.Setup(m => m.IsVersionLoaded(It.IsAny<string>(), It.IsAny<string>()))
                .Returns(true);
            MockVersionManager.Setup(m => m.GetLoadInfo(It.IsAny<string>(), It.IsAny<string>()))
                .Returns<string, string>((modelId, version) => new VersionLoadInfo
                {
                    ModelId = modelId,
                    Version = version,
                    IsLoaded = true,
                    MemoryUsageBytes = 1024 * 1024 * 100, // 100 MB
                    LoadTime = DateTime.UtcNow,
                    RequestCount = 0
                });

            // Setup router defaults
            MockRouter.Setup(r => r.GetDefaultVersion(It.IsAny<string>()))
                .Returns("v1.0.0");
            MockRouter.Setup(r => r.GetModel(It.IsAny<string>(), It.IsAny<RoutingContext>()))
                .Returns<string, RoutingContext>((modelName, context) => new Mock<IModel>().Object);
            MockRouter.Setup(r => r.GetModel(It.IsAny<string>(), It.IsAny<string>()))
                .Returns<string, string>((modelName, version) => new Mock<IModel>().Object);

            // Setup registry defaults
            MockRegistry.Setup(r => r.HasVersion(It.IsAny<string>(), It.IsAny<string>()))
                .Returns(true);
            MockRegistry.Setup(r => r.GetMetadata(It.IsAny<string>(), It.IsAny<string>()))
                .Returns<string, string>((name, version) => new ModelMetadata
                {
                    Version = version,
                    TrainingDate = DateTime.UtcNow.AddDays(-1),
                    ArtifactPath = $"/models/{name}/{version}/model.pt"
                });
        }

        private void InitializeTestData()
        {
            // Register test models
            MockRegistry.Object.RegisterModel("test-model", "v1.0.0", new ModelMetadata
            {
                Version = "v1.0.0",
                TrainingDate = DateTime.UtcNow.AddDays(-7),
                ArtifactPath = "/models/test-model/v1.0.0/model.pt"
            });

            MockRegistry.Object.RegisterModel("test-model", "v2.0.0", new ModelMetadata
            {
                Version = "v2.0.0",
                TrainingDate = DateTime.UtcNow.AddDays(-1),
                ArtifactPath = "/models/test-model/v2.0.0/model.pt"
            });

            MockRegistry.Object.RegisterModel("test-model", "v1.1.0", new ModelMetadata
            {
                Version = "v1.1.0",
                TrainingDate = DateTime.UtcNow.AddDays(-3),
                ArtifactPath = "/models/test-model/v1.1.0/model.pt"
            });
        }

        public void ResetMocks()
        {
            MockVersionManager.Reset();
            MockRouter.Reset();
            MockRegistry.Reset();
            SetupDefaultBehaviors();
        }

        public void Dispose()
        {
            // Cleanup resources
            MockVersionManager = null!;
            MockRouter = null!;
            MockRegistry = null!;
        }
    }
}
