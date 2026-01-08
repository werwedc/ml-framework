using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using MLFramework.Serving.Routing;
using MLFramework.Fusion.Backends;

namespace MLFramework.Tests.Serving.Routing
{
    /// <summary>
    /// Unit tests for HeaderRouter functionality.
    /// </summary>
    public class HeaderRouterTests
    {
        private readonly HeaderRouter _router;
        private readonly ILogger _logger;

        public HeaderRouterTests()
        {
            _logger = new ConsoleLogger(debugEnabled: true);
            _router = new HeaderRouter(_logger);
        }

        [Fact]
        public void RegisterExactMatchRule_RoutesMatchingRequests()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "2.0.0";
            var rule = new RoutingRule
            {
                HeaderName = "X-Feature",
                HeaderValue = "enabled",
                MatchType = MatchType.Exact,
                TargetVersion = version,
                Priority = 10
            };

            // Act
            _router.RegisterRoutingRule(modelName, rule);

            var context = new RoutingContext
            {
                Headers = new Dictionary<string, string>
                {
                    { "X-Feature", "enabled" }
                }
            };

            var result = _router.RouteByHeaders(modelName, context);

            // Assert
            Assert.Equal(version, result);
        }

        [Fact]
        public void RegisterPrefixMatchRule_RoutesPrefixes()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "2.0.0";
            var rule = new RoutingRule
            {
                HeaderName = "X-Feature",
                HeaderValue = "beta",
                MatchType = MatchType.Prefix,
                TargetVersion = version,
                Priority = 10
            };

            // Act
            _router.RegisterRoutingRule(modelName, rule);

            var context = new RoutingContext
            {
                Headers = new Dictionary<string, string>
                {
                    { "X-Feature", "beta-test-123" }
                }
            };

            var result = _router.RouteByHeaders(modelName, context);

            // Assert
            Assert.Equal(version, result);
        }

        [Fact]
        public void RegisterRegexRule_VerifiesPatternMatching()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "2.0.0";
            var rule = new RoutingRule
            {
                HeaderName = "X-User-Id",
                HeaderValue = @"^user-\d+$",
                MatchType = MatchType.Regex,
                TargetVersion = version,
                Priority = 10
            };

            // Act
            _router.RegisterRoutingRule(modelName, rule);

            var context = new RoutingContext
            {
                Headers = new Dictionary<string, string>
                {
                    { "X-User-Id", "user-12345" }
                }
            };

            var result = _router.RouteByHeaders(modelName, context);

            // Assert
            Assert.Equal(version, result);
        }

        [Fact]
        public void RegisterMultipleRulesWithPriorities_EvaluatesInOrder()
        {
            // Arrange
            const string modelName = "test-model";
            var rule1 = new RoutingRule
            {
                HeaderName = "X-Priority",
                HeaderValue = "high",
                MatchType = MatchType.Exact,
                TargetVersion = "3.0.0",
                Priority = 100
            };

            var rule2 = new RoutingRule
            {
                HeaderName = "X-Priority",
                HeaderValue = "high",
                MatchType = MatchType.Exact,
                TargetVersion = "2.0.0",
                Priority = 50
            };

            // Act
            _router.RegisterRoutingRule(modelName, rule1);
            _router.RegisterRoutingRule(modelName, rule2);

            var context = new RoutingContext
            {
                Headers = new Dictionary<string, string>
                {
                    { "X-Priority", "high" }
                }
            };

            var result = _router.RouteByHeaders(modelName, context);

            // Assert - Higher priority rule should match
            Assert.Equal("3.0.0", result);
        }

        [Fact]
        public void RegisterRuleWithEmptyHeaderName_ThrowsArgumentException()
        {
            // Arrange
            const string modelName = "test-model";
            var rule = new RoutingRule
            {
                HeaderName = "",
                HeaderValue = "value",
                MatchType = MatchType.Exact,
                TargetVersion = "2.0.0",
                Priority = 10
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.RegisterRoutingRule(modelName, rule));
        }

        [Fact]
        public void RegisterRuleWithEmptyHeaderValue_ThrowsArgumentException()
        {
            // Arrange
            const string modelName = "test-model";
            var rule = new RoutingRule
            {
                HeaderName = "X-Feature",
                HeaderValue = "",
                MatchType = MatchType.Exact,
                TargetVersion = "2.0.0",
                Priority = 10
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.RegisterRoutingRule(modelName, rule));
        }

        [Fact]
        public void RegisterRuleWithEmptyTargetVersion_ThrowsArgumentException()
        {
            // Arrange
            const string modelName = "test-model";
            var rule = new RoutingRule
            {
                HeaderName = "X-Feature",
                HeaderValue = "value",
                MatchType = MatchType.Exact,
                TargetVersion = "",
                Priority = 10
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.RegisterRoutingRule(modelName, rule));
        }

        [Fact]
        public void RegisterRuleWithInvalidRegex_ThrowsArgumentException()
        {
            // Arrange
            const string modelName = "test-model";
            var rule = new RoutingRule
            {
                HeaderName = "X-Feature",
                HeaderValue = "[invalid(regex",
                MatchType = MatchType.Regex,
                TargetVersion = "2.0.0",
                Priority = 10
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.RegisterRoutingRule(modelName, rule));
        }

        [Fact]
        public void RegisterNullRule_ThrowsArgumentNullException()
        {
            // Arrange
            const string modelName = "test-model";

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => _router.RegisterRoutingRule(modelName, null!));
        }

        [Fact]
        public void ContainsMatchRule_VerifiesContainsMatching()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "2.0.0";
            var rule = new RoutingRule
            {
                HeaderName = "X-Features",
                HeaderValue = "experimental",
                MatchType = MatchType.Contains,
                TargetVersion = version,
                Priority = 10
            };

            // Act
            _router.RegisterRoutingRule(modelName, rule);

            var context = new RoutingContext
            {
                Headers = new Dictionary<string, string>
                {
                    { "X-Features", "stable,experimental,beta" }
                }
            };

            var result = _router.RouteByHeaders(modelName, context);

            // Assert
            Assert.Equal(version, result);
        }

        [Fact]
        public void RouteWithNullHeaders_ReturnsNull()
        {
            // Arrange
            const string modelName = "test-model";
            var rule = new RoutingRule
            {
                HeaderName = "X-Feature",
                HeaderValue = "enabled",
                MatchType = MatchType.Exact,
                TargetVersion = "2.0.0",
                Priority = 10
            };

            // Act
            _router.RegisterRoutingRule(modelName, rule);

            var context = new RoutingContext
            {
                Headers = null
            };

            var result = _router.RouteByHeaders(modelName, context);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void RouteWithEmptyHeaders_ReturnsNull()
        {
            // Arrange
            const string modelName = "test-model";
            var rule = new RoutingRule
            {
                HeaderName = "X-Feature",
                HeaderValue = "enabled",
                MatchType = MatchType.Exact,
                TargetVersion = "2.0.0",
                Priority = 10
            };

            // Act
            _router.RegisterRoutingRule(modelName, rule);

            var context = new RoutingContext
            {
                Headers = new Dictionary<string, string>()
            };

            var result = _router.RouteByHeaders(modelName, context);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void RouteWithNoMatchingHeader_ReturnsNull()
        {
            // Arrange
            const string modelName = "test-model";
            var rule = new RoutingRule
            {
                HeaderName = "X-Feature",
                HeaderValue = "enabled",
                MatchType = MatchType.Exact,
                TargetVersion = "2.0.0",
                Priority = 10
            };

            // Act
            _router.RegisterRoutingRule(modelName, rule);

            var context = new RoutingContext
            {
                Headers = new Dictionary<string, string>
                {
                    { "X-Other-Header", "value" }
                }
            };

            var result = _router.RouteByHeaders(modelName, context);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void GetRules_ReturnsRulesSortedByPriority()
        {
            // Arrange
            const string modelName = "test-model";
            var rule1 = new RoutingRule
            {
                HeaderName = "X-1",
                HeaderValue = "value1",
                MatchType = MatchType.Exact,
                TargetVersion = "1.0.0",
                Priority = 10
            };

            var rule2 = new RoutingRule
            {
                HeaderName = "X-2",
                HeaderValue = "value2",
                MatchType = MatchType.Exact,
                TargetVersion = "2.0.0",
                Priority = 100
            };

            var rule3 = new RoutingRule
            {
                HeaderName = "X-3",
                HeaderValue = "value3",
                MatchType = MatchType.Exact,
                TargetVersion = "3.0.0",
                Priority = 50
            };

            // Act
            _router.RegisterRoutingRule(modelName, rule1);
            _router.RegisterRoutingRule(modelName, rule2);
            _router.RegisterRoutingRule(modelName, rule3);

            var rules = _router.GetRules(modelName).ToList();

            // Assert
            Assert.Equal(3, rules.Count);
            Assert.Equal(100, rules[0].Priority);
            Assert.Equal(50, rules[1].Priority);
            Assert.Equal(10, rules[2].Priority);
        }

        [Fact]
        public void GetRulesForNonExistentModel_ReturnsEmpty()
        {
            // Arrange
            const string modelName = "non-existent-model";

            // Act
            var rules = _router.GetRules(modelName);

            // Assert
            Assert.Empty(rules);
        }

        [Fact]
        public void ClearRules_RemovesAllRules()
        {
            // Arrange
            const string modelName = "test-model";
            var rule1 = new RoutingRule
            {
                HeaderName = "X-1",
                HeaderValue = "value1",
                MatchType = MatchType.Exact,
                TargetVersion = "1.0.0",
                Priority = 10
            };

            var rule2 = new RoutingRule
            {
                HeaderName = "X-2",
                HeaderValue = "value2",
                MatchType = MatchType.Exact,
                TargetVersion = "2.0.0",
                Priority = 20
            };

            // Act
            _router.RegisterRoutingRule(modelName, rule1);
            _router.RegisterRoutingRule(modelName, rule2);
            Assert.Equal(2, _router.GetRules(modelName).Count());

            _router.ClearRules(modelName);

            // Assert
            Assert.Empty(_router.GetRules(modelName));
        }

        [Fact]
        public void UnregisterRule_RemovesSpecificRule()
        {
            // Arrange
            const string modelName = "test-model";
            var rule1 = new RoutingRule
            {
                HeaderName = "X-1",
                HeaderValue = "value1",
                MatchType = MatchType.Exact,
                TargetVersion = "1.0.0",
                Priority = 10
            };

            var rule2 = new RoutingRule
            {
                HeaderName = "X-2",
                HeaderValue = "value2",
                MatchType = MatchType.Exact,
                TargetVersion = "2.0.0",
                Priority = 20
            };

            // Act
            _router.RegisterRoutingRule(modelName, rule1);
            _router.RegisterRoutingRule(modelName, rule2);

            var ruleId1 = rule1.Id;
            _router.UnregisterRoutingRule(modelName, ruleId1);

            // Assert
            var rules = _router.GetRules(modelName).ToList();
            Assert.Single(rules);
            Assert.Equal(rule2.Id, rules[0].Id);
        }

        [Fact]
        public void RouteWithEmptyModelName_ThrowsArgumentException()
        {
            // Arrange
            var context = new RoutingContext
            {
                Headers = new Dictionary<string, string>()
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.RouteByHeaders("", context));
        }

        [Fact]
        public void RegisterRuleWithEmptyModelName_ThrowsArgumentException()
        {
            // Arrange
            var rule = new RoutingRule
            {
                HeaderName = "X-Feature",
                HeaderValue = "value",
                MatchType = MatchType.Exact,
                TargetVersion = "2.0.0",
                Priority = 10
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.RegisterRoutingRule("", rule));
        }

        [Fact]
        public void Performance_HeaderBasedRoute_Under1ms()
        {
            // Arrange
            const string modelName = "test-model";
            var rule = new RoutingRule
            {
                HeaderName = "X-Feature",
                HeaderValue = "enabled",
                MatchType = MatchType.Exact,
                TargetVersion = "2.0.0",
                Priority = 10
            };

            _router.RegisterRoutingRule(modelName, rule);

            var context = new RoutingContext
            {
                Headers = new Dictionary<string, string>
                {
                    { "X-Feature", "enabled" }
                }
            };

            const int iterations = 1000;
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Act
            for (int i = 0; i < iterations; i++)
            {
                _router.RouteByHeaders(modelName, context);
            }

            stopwatch.Stop();
            var avgTimePerRoute = stopwatch.Elapsed.TotalMilliseconds / iterations;

            // Assert
            Assert.True(avgTimePerRoute < 1.0,
                $"Average routing time {avgTimePerRoute}ms exceeds target of 1.0ms");
        }

        [Fact]
        public void Performance_RegisterRoutingRule_Under1ms()
        {
            // Arrange
            const string modelName = "test-model";
            const int iterations = 100;

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Act
            for (int i = 0; i < iterations; i++)
            {
                var rule = new RoutingRule
                {
                    HeaderName = $"X-Feature-{i}",
                    HeaderValue = $"value-{i}",
                    MatchType = MatchType.Exact,
                    TargetVersion = $"2.0.{i}",
                    Priority = i
                };

                _router.RegisterRoutingRule(modelName, rule);
            }

            stopwatch.Stop();
            var avgTimePerRegistration = stopwatch.Elapsed.TotalMilliseconds / iterations;

            // Assert
            Assert.True(avgTimePerRegistration < 1.0,
                $"Average registration time {avgTimePerRegistration}ms exceeds target of 1.0ms");
        }

        [Fact]
        public async Task ConcurrentRuleRegistration_ThreadSafe()
        {
            // Arrange
            const string modelName = "test-model";
            const int threadCount = 100;

            var tasks = Enumerable.Range(0, threadCount).Select(i =>
                Task.Run(() =>
                {
                    var rule = new RoutingRule
                    {
                        HeaderName = $"X-Feature-{i}",
                        HeaderValue = $"value-{i}",
                        MatchType = MatchType.Exact,
                        TargetVersion = $"2.0.{i}",
                        Priority = i
                    };

                    _router.RegisterRoutingRule(modelName, rule);
                })
            );

            // Act
            await Task.WhenAll(tasks);

            // Assert
            var rules = _router.GetRules(modelName);
            Assert.Equal(threadCount, rules.Count());
        }
    }
}
