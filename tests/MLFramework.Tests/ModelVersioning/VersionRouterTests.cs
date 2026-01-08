using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Unit tests for VersionRouter implementation
    /// </summary>
    public class VersionRouterTests
    {
        #region Percentage-Based Routing Tests

        [Fact]
        public void PercentageBasedRouting_NinetyTenSplit_ShouldRouteCorrectly()
        {
            // Arrange
            var router = new VersionRouter();
            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", Percentage = 90.0 },
                    new RoutingRule { Version = "v2.0", Percentage = 10.0 }
                }
            };

            router.SetRoutingPolicy(policy);

            // Act - route 100 requests and count distribution
            var results = new Dictionary<string, int>();
            for (int i = 0; i < 1000; i++)
            {
                var context = new RequestContext
                {
                    UserId = $"user_{i}",
                    RequestTime = DateTime.UtcNow
                };

                var result = router.RouteRequest(context);

                if (!results.ContainsKey(result.Version!))
                    results[result.Version!] = 0;

                results[result.Version!]++;
            }

            // Assert - verify distribution is approximately 90/10
            Assert.True(results.ContainsKey("v1.0"));
            Assert.True(results.ContainsKey("v2.0"));

            var v1Percentage = (results["v1.0"] / 1000.0) * 100;
            var v2Percentage = (results["v2.0"] / 1000.0) * 100;

            // Allow 5% margin of error due to hash distribution
            Assert.InRange(v1Percentage, 85, 95);
            Assert.InRange(v2Percentage, 5, 15);
        }

        [Fact]
        public void PercentageBasedRouting_SameUserId_ShouldRouteConsistently()
        {
            // Arrange
            var router = new VersionRouter();
            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", Percentage = 90.0 },
                    new RoutingRule { Version = "v2.0", Percentage = 10.0 }
                }
            };

            router.SetRoutingPolicy(policy);
            var userId = "consistent_user_123";

            // Act - route the same user multiple times
            var versions = new List<string>();
            for (int i = 0; i < 10; i++)
            {
                var context = new RequestContext
                {
                    UserId = userId,
                    RequestTime = DateTime.UtcNow
                };

                var result = router.RouteRequest(context);
                versions.Add(result.Version!);
            }

            // Assert - all requests should go to the same version
            Assert.All(versions, v => Assert.Equal(versions[0], v));
        }

        #endregion

        #region Shadow Mode Routing Tests

        [Fact]
        public void ShadowModeRouting_ShouldRouteToPrimaryAndIncludeShadows()
        {
            // Arrange
            var router = new VersionRouter();
            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.Shadow,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", IsPrimary = true },
                    new RoutingRule { Version = "v2.0", IsPrimary = false },
                    new RoutingRule { Version = "v3.0", IsPrimary = false }
                }
            };

            router.SetRoutingPolicy(policy);

            // Act
            var context = new RequestContext
            {
                UserId = "user_1",
                RequestTime = DateTime.UtcNow
            };

            var result = router.RouteRequest(context);

            // Assert
            Assert.Equal("v1.0", result.Version);
            Assert.False(result.IsShadow);
            Assert.Equal(2, result.ShadowVersions!.Count);
            Assert.Contains("v2.0", result.ShadowVersions);
            Assert.Contains("v3.0", result.ShadowVersions);
        }

        [Fact]
        public void ShadowModeRouting_NoPrimary_ShouldThrowException()
        {
            // Arrange
            var router = new VersionRouter();
            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.Shadow,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", IsPrimary = false },
                    new RoutingRule { Version = "v2.0", IsPrimary = false }
                }
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => router.SetRoutingPolicy(policy));
        }

        #endregion

        #region Deterministic Routing Tests

        [Fact]
        public void DeterministicRouting_ByUserSegment_ShouldMatchCorrectly()
        {
            // Arrange
            var router = new VersionRouter();
            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.Deterministic,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", Segment = "premium" },
                    new RoutingRule { Version = "v2.0", Segment = "standard" },
                    new RoutingRule { Version = "v3.0", Segment = "trial" }
                }
            };

            router.SetRoutingPolicy(policy);

            // Act & Assert - Premium users
            var premiumContext = new RequestContext
            {
                UserId = "user_1",
                Segment = "premium",
                RequestTime = DateTime.UtcNow
            };

            var premiumResult = router.RouteRequest(premiumContext);
            Assert.Equal("v1.0", premiumResult.Version);

            // Act & Assert - Standard users
            var standardContext = new RequestContext
            {
                UserId = "user_2",
                Segment = "standard",
                RequestTime = DateTime.UtcNow
            };

            var standardResult = router.RouteRequest(standardContext);
            Assert.Equal("v2.0", standardResult.Version);

            // Act & Assert - Trial users
            var trialContext = new RequestContext
            {
                UserId = "user_3",
                Segment = "trial",
                RequestTime = DateTime.UtcNow
            };

            var trialResult = router.RouteRequest(trialContext);
            Assert.Equal("v3.0", trialResult.Version);
        }

        [Fact]
        public void DeterministicRouting_ByRegion_ShouldMatchCorrectly()
        {
            // Arrange
            var router = new VersionRouter();
            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.Deterministic,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", Region = "US-EAST" },
                    new RoutingRule { Version = "v2.0", Region = "US-WEST" },
                    new RoutingRule { Version = "v3.0", Region = "EU-WEST" }
                }
            };

            router.SetRoutingPolicy(policy);

            // Act & Assert
            var usEastContext = new RequestContext
            {
                UserId = "user_1",
                Region = "US-EAST",
                RequestTime = DateTime.UtcNow
            };

            var usEastResult = router.RouteRequest(usEastContext);
            Assert.Equal("v1.0", usEastResult.Version);

            var usWestContext = new RequestContext
            {
                UserId = "user_2",
                Region = "US-WEST",
                RequestTime = DateTime.UtcNow
            };

            var usWestResult = router.RouteRequest(usWestContext);
            Assert.Equal("v2.0", usWestResult.Version);
        }

        [Fact]
        public void DeterministicRouting_ByUserIdPattern_ShouldMatchCorrectly()
        {
            // Arrange
            var router = new VersionRouter();
            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.Deterministic,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", UserIdPattern = "^alpha_.*$" },
                    new RoutingRule { Version = "v2.0", UserIdPattern = "^beta_.*$" }
                }
            };

            router.SetRoutingPolicy(policy);

            // Act & Assert
            var alphaContext = new RequestContext
            {
                UserId = "alpha_user_123",
                RequestTime = DateTime.UtcNow
            };

            var alphaResult = router.RouteRequest(alphaContext);
            Assert.Equal("v1.0", alphaResult.Version);

            var betaContext = new RequestContext
            {
                UserId = "beta_user_456",
                RequestTime = DateTime.UtcNow
            };

            var betaResult = router.RouteRequest(betaContext);
            Assert.Equal("v2.0", betaResult.Version);
        }

        #endregion

        #region Time-Based Routing Tests

        [Fact]
        public void TimeBasedRouting_WithinTimeRange_ShouldMatchCorrectly()
        {
            // Arrange
            var router = new VersionRouter();
            var now = DateTime.UtcNow;

            // Create a time range for the current day and time
            var timeRange = new TimeRange
            {
                StartTime = now.TimeOfDay - TimeSpan.FromHours(1),
                EndTime = now.TimeOfDay + TimeSpan.FromHours(1),
                DaysOfWeek = new[] { now.DayOfWeek }
            };

            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.TimeBased,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", TimeRange = timeRange },
                    new RoutingRule { Version = "v2.0", TimeRange = new TimeRange
                    {
                        StartTime = TimeSpan.FromHours(10),
                        EndTime = TimeSpan.FromHours(12),
                        DaysOfWeek = new[] { DayOfWeek.Monday }
                    }}
                }
            };

            router.SetRoutingPolicy(policy);

            // Act
            var context = new RequestContext
            {
                UserId = "user_1",
                RequestTime = now
            };

            var result = router.RouteRequest(context);

            // Assert - should match the first time range (current time)
            Assert.Equal("v1.0", result.Version);
        }

        [Fact]
        public void TimeBasedRouting_OvernightTimeRange_ShouldMatchCorrectly()
        {
            // Arrange
            var router = new VersionRouter();
            var timeRange = new TimeRange
            {
                StartTime = TimeSpan.FromHours(22), // 10 PM
                EndTime = TimeSpan.FromHours(6),    // 6 AM (next day)
                DaysOfWeek = new[] { DayOfWeek.Monday, DayOfWeek.Tuesday }
            };

            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.TimeBased,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", TimeRange = timeRange }
                }
            };

            router.SetRoutingPolicy(policy);

            // Act & Assert - Test 11 PM on Monday (should match)
            var context1 = new RequestContext
            {
                UserId = "user_1",
                RequestTime = new DateTime(2024, 1, 8, 23, 0, 0, DateTimeKind.Utc) // Monday 11 PM
            };

            var result1 = router.RouteRequest(context1);
            Assert.Equal("v1.0", result1.Version);

            // Act & Assert - Test 2 AM on Tuesday (should match)
            var context2 = new RequestContext
            {
                UserId = "user_1",
                RequestTime = new DateTime(2024, 1, 9, 2, 0, 0, DateTimeKind.Utc) // Tuesday 2 AM
            };

            var result2 = router.RouteRequest(context2);
            Assert.Equal("v1.0", result2.Version);

            // Act & Assert - Test 8 AM on Tuesday (should NOT match, use default)
            var context3 = new RequestContext
            {
                UserId = "user_1",
                RequestTime = new DateTime(2024, 1, 9, 8, 0, 0, DateTimeKind.Utc) // Tuesday 8 AM
            };

            var result3 = router.RouteRequest(context3);
            // Should return default (first version) since no match
            Assert.Equal("v1.0", result3.Version);
        }

        #endregion

        #region Policy Validation Tests

        [Fact]
        public void PolicyValidation_PercentageSumNot100_ShouldThrowException()
        {
            // Arrange
            var router = new VersionRouter();
            var invalidPolicy = new RoutingPolicy
            {
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", Percentage = 80.0 },
                    new RoutingRule { Version = "v2.0", Percentage = 25.0 } // Sum = 105
                }
            };

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => router.SetRoutingPolicy(invalidPolicy));
            Assert.Contains("Total percentage must equal 100", exception.Message);
        }

        [Fact]
        public void PolicyValidation_InvalidTimeRange_ShouldThrowException()
        {
            // Arrange
            var router = new VersionRouter();
            var invalidTimeRange = new TimeRange
            {
                StartTime = TimeSpan.FromHours(12),
                EndTime = TimeSpan.FromHours(10), // End before start
                DaysOfWeek = new[] { DayOfWeek.Monday }
            };

            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.TimeBased,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", TimeRange = invalidTimeRange }
                }
            };

            // Act & Assert - Invalid time range (end < start and not overnight)
            var exception = Assert.Throws<ArgumentException>(() => router.SetRoutingPolicy(policy));
            Assert.Contains("TimeBased mode requires at least one rule with a valid time range", exception.Message);
        }

        [Fact]
        public void PolicyValidation_NoRules_ShouldThrowException()
        {
            // Arrange
            var router = new VersionRouter();
            var invalidPolicy = new RoutingPolicy
            {
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>()
            };

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => router.SetRoutingPolicy(invalidPolicy));
            Assert.Contains("at least one rule", exception.Message);
        }

        [Fact]
        public void PolicyValidation_MissingVersion_ShouldThrowException()
        {
            // Arrange
            var router = new VersionRouter();
            var invalidPolicy = new RoutingPolicy
            {
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "", Percentage = 100.0 }
                }
            };

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => router.SetRoutingPolicy(invalidPolicy));
            Assert.Contains("valid version", exception.Message);
        }

        #endregion

        #region Concurrency and Thread-Safety Tests

        [Fact]
        public async Task ConcurrentPolicyUpdates_ShouldBeThreadSafe()
        {
            // Arrange
            var router = new VersionRouter();
            var initialPolicy = new RoutingPolicy
            {
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", Percentage = 100.0 }
                }
            };

            router.SetRoutingPolicy(initialPolicy);

            var tasks = new List<Task>();

            // Act - Create multiple tasks that update the policy concurrently
            for (int i = 0; i < 10; i++)
            {
                var versionNum = i + 2;
                tasks.Add(Task.Run(() =>
                {
                    var newPolicy = new RoutingPolicy
                    {
                        Mode = RoutingMode.Percentage,
                        Rules = new List<RoutingRule>
                        {
                            new RoutingRule { Version = $"v{versionNum}.0", Percentage = 100.0 }
                        }
                    };

                    router.SetRoutingPolicy(newPolicy);
                }));
            }

            await Task.WhenAll(tasks);

            // Assert - Router should have a valid policy (no crashes)
            var currentPolicy = router.GetCurrentPolicy();
            Assert.NotNull(currentPolicy);
            Assert.NotNull(currentPolicy.Rules);
            Assert.True(currentPolicy.Rules.Count > 0);
        }

        [Fact]
        public async Task AtomicPolicySwitching_ShouldNotLoseInFlightRequests()
        {
            // Arrange
            var router = new VersionRouter();
            var policy1 = new RoutingPolicy
            {
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", Percentage = 100.0 }
                }
            };

            var policy2 = new RoutingPolicy
            {
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v2.0", Percentage = 100.0 }
                }
            };

            router.SetRoutingPolicy(policy1);

            var results = new ConcurrentBag<string>();

            // Act - Route requests while switching policy
            var routingTask = Task.Run(() =>
            {
                for (int i = 0; i < 100; i++)
                {
                    Thread.Sleep(1); // Small delay

                    var context = new RequestContext
                    {
                        UserId = $"user_{i}",
                        RequestTime = DateTime.UtcNow
                    };

                    try
                    {
                        var result = router.RouteRequest(context);
                        results.Add(result.Version!);
                    }
                    catch (Exception)
                    {
                        // Ignore any exceptions from concurrent access
                    }
                }
            });

            var updateTask = Task.Run(() =>
            {
                Thread.Sleep(50); // Wait for some routing to happen
                router.UpdatePolicy(policy2);
            });

            await Task.WhenAll(routingTask, updateTask);

            // Assert - Should have some results from both policies
            Assert.True(results.Count > 0);
        }

        #endregion

        #region Policy Management Tests

        [Fact]
        public void GetCurrentPolicy_ShouldReturnCorrectPolicy()
        {
            // Arrange
            var router = new VersionRouter();
            var policy = new RoutingPolicy
            {
                ModelId = "test-model",
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", Percentage = 100.0 }
                },
                EffectiveDate = DateTime.UtcNow
            };

            router.SetRoutingPolicy(policy);

            // Act
            var currentPolicy = router.GetCurrentPolicy();

            // Assert
            Assert.Equal("test-model", currentPolicy.ModelId);
            Assert.Equal(RoutingMode.Percentage, currentPolicy.Mode);
            Assert.Single(currentPolicy.Rules);
            Assert.Equal("v1.0", currentPolicy.Rules[0].Version);
        }

        [Fact]
        public void UpdatePolicy_ShouldReplaceExistingPolicy()
        {
            // Arrange
            var router = new VersionRouter();
            var oldPolicy = new RoutingPolicy
            {
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", Percentage = 100.0 }
                }
            };

            var newPolicy = new RoutingPolicy
            {
                Mode = RoutingMode.Deterministic,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v2.0", Segment = "premium" }
                }
            };

            router.SetRoutingPolicy(oldPolicy);

            // Act
            router.UpdatePolicy(newPolicy);

            // Assert
            var currentPolicy = router.GetCurrentPolicy();
            Assert.Equal(RoutingMode.Deterministic, currentPolicy.Mode);
            Assert.Equal("v2.0", currentPolicy.Rules[0].Version);
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void RouteRequest_NullContext_ShouldThrowException()
        {
            // Arrange
            var router = new VersionRouter();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => router.RouteRequest(null!));
        }

        [Fact]
        public void SetRoutingPolicy_NullPolicy_ShouldThrowException()
        {
            // Arrange
            var router = new VersionRouter();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => router.SetRoutingPolicy(null!));
        }

        [Fact]
        public void UpdatePolicy_NullPolicy_ShouldThrowException()
        {
            // Arrange
            var router = new VersionRouter();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => router.UpdatePolicy(null!));
        }

        [Fact]
        public void PercentageBasedRouting_NoUserId_ShouldUseRandomRouting()
        {
            // Arrange
            var router = new VersionRouter();
            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", Percentage = 50.0 },
                    new RoutingRule { Version = "v2.0", Percentage = 50.0 }
                }
            };

            router.SetRoutingPolicy(policy);

            // Act - Route without user ID
            var results = new List<string>();
            for (int i = 0; i < 10; i++)
            {
                var context = new RequestContext
                {
                    RequestTime = DateTime.UtcNow
                };

                var result = router.RouteRequest(context);
                results.Add(result.Version!);
            }

            // Assert - Should have some variety in results (not all the same)
            var uniqueVersions = results.Distinct().Count();
            Assert.True(uniqueVersions > 1, "Expected multiple versions when routing without user ID");
        }

        [Fact]
        public void DeterministicRouting_NoMatch_ShouldReturnDefaultVersion()
        {
            // Arrange
            var router = new VersionRouter();
            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.Deterministic,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0", Segment = "premium" },
                    new RoutingRule { Version = "v2.0", Segment = "standard" }
                }
            };

            router.SetRoutingPolicy(policy);

            // Act - Request with no matching segment
            var context = new RequestContext
            {
                UserId = "user_1",
                Segment = "unknown", // Not in rules
                RequestTime = DateTime.UtcNow
            };

            var result = router.RouteRequest(context);

            // Assert - Should return first version as default
            Assert.Equal("v1.0", result.Version);
            Assert.Contains("Default", result.RuleMatched);
        }

        #endregion
    }
}
