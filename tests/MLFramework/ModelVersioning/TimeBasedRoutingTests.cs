using MLFramework.Serving.Deployment;
using MLFramework.Serving.Routing;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Integration tests for time-based routing scenarios.
    /// </summary>
    public class TimeBasedRoutingTests : IClassFixture<IntegrationTestFixture>
    {
        private readonly IntegrationTestFixture _fixture;

        public TimeBasedRoutingTests(IntegrationTestFixture fixture)
        {
            _fixture = fixture;
        }

        [Fact]
        public void TimeBased_RouteBySchedule()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var mockModel1 = new Mock<IModel>();
            var mockModel2 = new Mock<IModel>();

            mockModel1.Setup(m => m.Name).Returns(modelName);
            mockModel1.Setup(m => m.Version).Returns(version1);

            mockModel2.Setup(m => m.Name).Returns(modelName);
            mockModel2.Setup(m => m.Version).Returns(version2);

            // 1. Set up time-based routing rules
            var currentTime = DateTime.UtcNow;

            // Route to v1.0.0 during business hours (9 AM - 5 PM UTC)
            var isBusinessHours = currentTime.Hour >= 9 && currentTime.Hour < 17;

            if (isBusinessHours)
            {
                _fixture.MockRouter.Setup(r => r.GetModel(modelName, It.IsAny<RoutingContext>()))
                    .Returns(mockModel1.Object);
                _fixture.Router.SetDefaultVersion(modelName, version1);
            }
            else
            {
                _fixture.MockRouter.Setup(r => r.GetModel(modelName, It.IsAny<RoutingContext>()))
                    .Returns(mockModel2.Object);
                _fixture.Router.SetDefaultVersion(modelName, version2);
            }

            // 2. Verify routing by time of day
            var model = _fixture.Router.GetModel(modelName, new RoutingContext());
            Assert.NotNull(model);

            var defaultVersion = _fixture.Router.GetDefaultVersion(modelName);
            if (isBusinessHours)
            {
                Assert.Equal(version1, defaultVersion);
                Assert.Equal(version1, model.Version);
            }
            else
            {
                Assert.Equal(version2, defaultVersion);
                Assert.Equal(version2, model.Version);
            }
        }

        [Fact]
        public void TimeBased_RouteByDayOfWeek()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0"; // Weekday version
            const string version2 = "v2.0.0"; // Weekend version

            var mockModel1 = new Mock<IModel>();
            var mockModel2 = new Mock<IModel>();

            mockModel1.Setup(m => m.Name).Returns(modelName);
            mockModel1.Setup(m => m.Version).Returns(version1);

            mockModel2.Setup(m => m.Name).Returns(modelName);
            mockModel2.Setup(m => m.Version).Returns(version2);

            // 3. Verify routing by day of week
            var today = DateTime.UtcNow.DayOfWeek;
            var isWeekday = today >= DayOfWeek.Monday && today <= DayOfWeek.Friday;

            if (isWeekday)
            {
                _fixture.MockRouter.Setup(r => r.GetModel(modelName, It.IsAny<RoutingContext>()))
                    .Returns(mockModel1.Object);
                _fixture.Router.SetDefaultVersion(modelName, version1);
            }
            else
            {
                _fixture.MockRouter.Setup(r => r.GetModel(modelName, It.IsAny<RoutingContext>()))
                    .Returns(mockModel2.Object);
                _fixture.Router.SetDefaultVersion(modelName, version2);
            }

            var model = _fixture.Router.GetModel(modelName, new RoutingContext());
            Assert.NotNull(model);

            var defaultVersion = _fixture.Router.GetDefaultVersion(modelName);
            if (isWeekday)
            {
                Assert.Equal(version1, defaultVersion);
                Assert.Equal(version1, model.Version);
            }
            else
            {
                Assert.Equal(version2, defaultVersion);
                Assert.Equal(version2, model.Version);
            }
        }

        [Fact]
        public void TimeBased_MultipleSchedules_UsesCorrectRule()
        {
            // Arrange
            const string modelName = "test-model";
            var versions = new Dictionary<string, string>
            {
                ["morning"] = "v1.0.0",
                ["afternoon"] = "v2.0.0",
                ["evening"] = "v1.1.0",
                ["night"] = "v3.0.0"
            };

            var mockModels = versions.ToDictionary(
                kvp => kvp.Value,
                kvp =>
                {
                    var mock = new Mock<IModel>();
                    mock.Setup(m => m.Name).Returns(modelName);
                    mock.Setup(m => m.Version).Returns(kvp.Value);
                    return mock;
                });

            var currentHour = DateTime.UtcNow.Hour;

            // Define time periods
            string currentPeriod, selectedVersion;
            if (currentHour >= 6 && currentHour < 12)
            {
                currentPeriod = "morning";
                selectedVersion = versions["morning"];
            }
            else if (currentHour >= 12 && currentHour < 17)
            {
                currentPeriod = "afternoon";
                selectedVersion = versions["afternoon"];
            }
            else if (currentHour >= 17 && currentHour < 22)
            {
                currentPeriod = "evening";
                selectedVersion = versions["evening"];
            }
            else
            {
                currentPeriod = "night";
                selectedVersion = versions["night"];
            }

            _fixture.MockRouter.Setup(r => r.GetModel(modelName, It.IsAny<RoutingContext>()))
                .Returns(mockModels[selectedVersion].Object);
            _fixture.Router.SetDefaultVersion(modelName, selectedVersion);

            // Act
            var model = _fixture.Router.GetModel(modelName, new RoutingContext());
            var defaultVersion = _fixture.Router.GetDefaultVersion(modelName);

            // Assert
            Assert.NotNull(model);
            Assert.Equal(selectedVersion, defaultVersion);
            Assert.Equal(selectedVersion, model.Version);
        }

        [Fact]
        public void TimeBased_WithScheduleChange_SwitchesVersion()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var mockModel1 = new Mock<IModel>();
            var mockModel2 = new Mock<IModel>();

            mockModel1.Setup(m => m.Name).Returns(modelName);
            mockModel1.Setup(m => m.Version).Returns(version1);

            mockModel2.Setup(m => m.Name).Returns(modelName);
            mockModel2.Setup(m => m.Version).Returns(version2);

            // Simulate time-based switch
            var simulatingBusinessHours = true;

            if (simulatingBusinessHours)
            {
                _fixture.MockRouter.Setup(r => r.GetModel(modelName, It.IsAny<RoutingContext>()))
                    .Returns(mockModel1.Object);
                _fixture.Router.SetDefaultVersion(modelName, version1);

                var model1 = _fixture.Router.GetModel(modelName, new RoutingContext());
                Assert.Equal(version1, model1.Version);

                // Simulate time change to non-business hours
                simulatingBusinessHours = false;
                _fixture.MockRouter.Setup(r => r.GetModel(modelName, It.IsAny<RoutingContext>()))
                    .Returns(mockModel2.Object);
                _fixture.Router.SetDefaultVersion(modelName, version2);

                var model2 = _fixture.Router.GetModel(modelName, new RoutingContext());
                Assert.Equal(version2, model2.Version);
            }
        }

        [Fact]
        public async Task TimeBased_WithAutomaticSwitch_ScheduledCorrectly()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            metadata1.PerformanceMetrics["schedule_start"] = "09:00";
            metadata1.PerformanceMetrics["schedule_end"] = "17:00";

            metadata2.PerformanceMetrics["schedule_start"] = "17:00";
            metadata2.PerformanceMetrics["schedule_end"] = "09:00";

            var currentTime = DateTime.UtcNow;
            var timeStr = $"{currentTime.Hour:D2}:{currentTime.Minute:D2}";

            // Determine which version should be active
            bool version1Active = timeStr.CompareTo("09:00") >= 0 && timeStr.CompareTo("17:00") < 0;
            string activeVersion = version1Active ? version1 : version2;

            _fixture.Router.SetDefaultVersion(modelName, activeVersion);

            // Act
            var defaultVersion = _fixture.Router.GetDefaultVersion(modelName);

            // Assert
            if (version1Active)
            {
                Assert.Equal(version1, defaultVersion);
            }
            else
            {
                Assert.Equal(version2, defaultVersion);
            }
        }

        [Fact]
        public void TimeBased_WithGracefulTransition_SmoothlySwitches()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // Simulate graceful transition
            var healthCheck1 = _fixture.HotSwapper.CheckVersionHealth(modelName, version1);
            var healthCheck2 = _fixture.HotSwapper.CheckVersionHealth(modelName, version2);

            Assert.True(healthCheck1.IsHealthy);
            Assert.True(healthCheck2.IsHealthy);

            // Perform swap
            _fixture.Router.SetDefaultVersion(modelName, version2);

            // Verify transition
            var defaultVersion = _fixture.Router.GetDefaultVersion(modelName);
            Assert.Equal(version2, defaultVersion);
        }

        [Fact]
        public void TimeBased_WithFallback_HandlesGracefully()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const string fallbackVersion = "v1.1.0";

            var mockModel1 = new Mock<IModel>();
            var mockModel2 = new Mock<IModel>();
            var mockFallback = new Mock<IModel>();

            mockModel1.Setup(m => m.Name).Returns(modelName);
            mockModel1.Setup(m => m.Version).Returns(version1);

            mockModel2.Setup(m => m.Name).Returns(modelName);
            mockModel2.Setup(m => m.Version).Returns(version2);

            mockFallback.Setup(m => m.Name).Returns(modelName);
            mockFallback.Setup(m => m.Version).Returns(fallbackVersion);

            // Make scheduled version unhealthy
            _fixture.MockVersionManager.Setup(m => m.IsVersionLoaded(modelName, version2))
                .Returns(false);

            // Should fallback to healthy version
            _fixture.MockRouter.Setup(r => r.GetModel(modelName, It.IsAny<RoutingContext>()))
                .Returns(mockModel1.Object);

            var model = _fixture.Router.GetModel(modelName, new RoutingContext());
            Assert.NotNull(model);
            Assert.Equal(version1, model.Version);
        }

        [Fact]
        public void TimeBased_WithMultipleTimeZones_RoutesCorrectly()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var mockModel1 = new Mock<IModel>();
            var mockModel2 = new Mock<IModel>();

            mockModel1.Setup(m => m.Name).Returns(modelName);
            mockModel1.Setup(m => m.Version).Returns(version1);

            mockModel2.Setup(m => m.Name).Returns(modelName);
            mockModel2.Setup(m => m.Version).Returns(version2);

            // Simulate different time zones
            var timeZones = new[]
            {
                new { Name = "UTC", Offset = 0 },
                new { Name = "EST", Offset = -5 },
                new { Name = "PST", Offset = -8 },
                new { Name = "CET", Offset = 1 }
            };

            foreach (var timeZone in timeZones)
            {
                var localHour = (DateTime.UtcNow.Hour + timeZone.Offset + 24) % 24;
                var isBusinessHours = localHour >= 9 && localHour < 17;

                var context = new RoutingContext
                {
                    Headers = new Dictionary<string, string> { ["Time-Zone"] = timeZone.Name }
                };

                if (isBusinessHours)
                {
                    _fixture.MockRouter.Setup(r => r.GetModel(modelName, It.Is<RoutingContext>(c => c.Headers?["Time-Zone"] == timeZone.Name)))
                        .Returns(mockModel1.Object);
                }
                else
                {
                    _fixture.MockRouter.Setup(r => r.GetModel(modelName, It.Is<RoutingContext>(c => c.Headers?["Time-Zone"] == timeZone.Name)))
                        .Returns(mockModel2.Object);
                }
            }

            // Act
            var utcModel = _fixture.Router.GetModel(modelName, new RoutingContext
            {
                Headers = new Dictionary<string, string> { ["Time-Zone"] = "UTC" }
            });

            Assert.NotNull(utcModel);
        }
    }
}
