using System.Text.Json;
using MLFramework.ModelVersioning;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    public class RoutingDataModelsTests
    {
        [Fact]
        public void RoutingPolicy_CreationWithMultipleRules_Succeeds()
        {
            // Arrange & Act
            var policy = new RoutingPolicy
            {
                ModelId = "test-model",
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0.0", Percentage = 90 },
                    new RoutingRule { Version = "v2.0.0", Percentage = 10 }
                }
            };

            // Assert
            Assert.Equal("test-model", policy.ModelId);
            Assert.Equal(RoutingMode.Percentage, policy.Mode);
            Assert.Equal(2, policy.Rules?.Count);
        }

        [Fact]
        public void RoutingRule_PercentageBounds_ValidatesCorrectly()
        {
            // Arrange & Act
            var rule1 = new RoutingRule { Version = "v1.0.0", Percentage = 0 };
            var rule2 = new RoutingRule { Version = "v2.0.0", Percentage = 50 };
            var rule3 = new RoutingRule { Version = "v3.0.0", Percentage = 100 };

            // Assert
            Assert.Equal(0, rule1.Percentage);
            Assert.Equal(50, rule2.Percentage);
            Assert.Equal(100, rule3.Percentage);
        }

        [Fact]
        public void TimeRange_Validation_StartLessThanEnd()
        {
            // Arrange
            var timeRange = new TimeRange
            {
                StartTime = new TimeSpan(9, 0, 0),
                EndTime = new TimeSpan(17, 0, 0),
                DaysOfWeek = new[] { DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday, DayOfWeek.Thursday, DayOfWeek.Friday }
            };

            // Act
            bool isValid = timeRange.IsValid();

            // Assert
            Assert.True(isValid);
        }

        [Fact]
        public void TimeRange_Validation_InvalidWithoutDays()
        {
            // Arrange
            var timeRange = new TimeRange
            {
                StartTime = new TimeSpan(9, 0, 0),
                EndTime = new TimeSpan(17, 0, 0),
                DaysOfWeek = Array.Empty<DayOfWeek>()
            };

            // Act
            bool isValid = timeRange.IsValid();

            // Assert
            Assert.False(isValid);
        }

        [Fact]
        public void TimeRange_Validation_InvalidWhenStartGreaterThanEnd()
        {
            // Arrange
            var timeRange = new TimeRange
            {
                StartTime = new TimeSpan(17, 0, 0),
                EndTime = new TimeSpan(9, 0, 0),
                DaysOfWeek = new[] { DayOfWeek.Monday }
            };

            // Act
            bool isValid = timeRange.IsValid();

            // Assert
            Assert.False(isValid);
        }

        [Fact]
        public void TimeRange_IsInRange_ReturnsTrueForValidTime()
        {
            // Arrange
            var timeRange = new TimeRange
            {
                StartTime = new TimeSpan(9, 0, 0),
                EndTime = new TimeSpan(17, 0, 0),
                DaysOfWeek = new[] { DayOfWeek.Monday, DayOfWeek.Tuesday }
            };

            var testTime = new DateTime(2024, 1, 15, 14, 30, 0); // Monday at 2:30 PM

            // Act
            bool inRange = timeRange.IsInRange(testTime);

            // Assert
            Assert.True(inRange);
        }

        [Fact]
        public void TimeRange_IsInRange_ReturnsFalseForWrongDay()
        {
            // Arrange
            var timeRange = new TimeRange
            {
                StartTime = new TimeSpan(9, 0, 0),
                EndTime = new TimeSpan(17, 0, 0),
                DaysOfWeek = new[] { DayOfWeek.Monday, DayOfWeek.Tuesday }
            };

            var testTime = new DateTime(2024, 1, 20, 14, 30, 0); // Saturday at 2:30 PM

            // Act
            bool inRange = timeRange.IsInRange(testTime);

            // Assert
            Assert.False(inRange);
        }

        [Fact]
        public void TimeRange_IsInRange_ReturnsFalseForWrongTime()
        {
            // Arrange
            var timeRange = new TimeRange
            {
                StartTime = new TimeSpan(9, 0, 0),
                EndTime = new TimeSpan(17, 0, 0),
                DaysOfWeek = new[] { DayOfWeek.Monday }
            };

            var testTime = new DateTime(2024, 1, 15, 18, 30, 0); // Monday at 6:30 PM

            // Act
            bool inRange = timeRange.IsInRange(testTime);

            // Assert
            Assert.False(inRange);
        }

        [Fact]
        public void TimeRange_IsInRange_OvernightRange_WorksCorrectly()
        {
            // Arrange
            var timeRange = new TimeRange
            {
                StartTime = new TimeSpan(22, 0, 0),
                EndTime = new TimeSpan(6, 0, 0),
                DaysOfWeek = new[] { DayOfWeek.Monday }
            };

            var testTime1 = new DateTime(2024, 1, 15, 23, 0, 0); // Monday at 11:00 PM
            var testTime2 = new DateTime(2024, 1, 15, 3, 0, 0); // Monday at 3:00 AM
            var testTime3 = new DateTime(2024, 1, 15, 12, 0, 0); // Monday at 12:00 PM

            // Act
            bool inRange1 = timeRange.IsInRange(testTime1);
            bool inRange2 = timeRange.IsInRange(testTime2);
            bool inRange3 = timeRange.IsInRange(testTime3);

            // Assert
            Assert.True(inRange1);
            Assert.True(inRange2);
            Assert.False(inRange3);
        }

        [Fact]
        public void RequestContext_CreationWithMetadata_Succeeds()
        {
            // Arrange & Act
            var context = new RequestContext
            {
                UserId = "user123",
                Segment = "premium",
                Region = "us-east-1",
                Metadata = new Dictionary<string, string>
                {
                    { "device", "mobile" },
                    { "appVersion", "2.1.0" }
                }
            };

            // Assert
            Assert.Equal("user123", context.UserId);
            Assert.Equal("premium", context.Segment);
            Assert.Equal("us-east-1", context.Region);
            Assert.Equal("mobile", context.Metadata["device"]);
            Assert.Equal("2.1.0", context.Metadata["appVersion"]);
        }

        [Fact]
        public void RoutingResult_Construction_Succeeds()
        {
            // Arrange & Act
            var result = new RoutingResult
            {
                Version = "v1.0.0",
                IsShadow = false,
                ShadowVersions = new List<string>(),
                RuleMatched = "Percentage split: 90%"
            };

            // Assert
            Assert.Equal("v1.0.0", result.Version);
            Assert.False(result.IsShadow);
            Assert.Empty(result.ShadowVersions);
            Assert.Equal("Percentage split: 90%", result.RuleMatched);
        }

        [Fact]
        public void RoutingPolicy_JsonSerialization_RoundtripSucceeds()
        {
            // Arrange
            var originalPolicy = new RoutingPolicy
            {
                ModelId = "test-model",
                Mode = RoutingMode.Percentage,
                EffectiveDate = DateTime.UtcNow,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0.0", Percentage = 90 },
                    new RoutingRule { Version = "v2.0.0", Percentage = 10 }
                }
            };

            // Act
            string json = JsonSerializer.Serialize(originalPolicy);
            var deserializedPolicy = JsonSerializer.Deserialize<RoutingPolicy>(json);

            // Assert
            Assert.NotNull(deserializedPolicy);
            Assert.Equal(originalPolicy.ModelId, deserializedPolicy.ModelId);
            Assert.Equal(originalPolicy.Mode, deserializedPolicy.Mode);
            Assert.Equal(originalPolicy.Rules?.Count, deserializedPolicy.Rules?.Count);
            Assert.Equal(originalPolicy.Rules?[0].Version, deserializedPolicy.Rules?[0].Version);
            Assert.Equal(originalPolicy.Rules?[0].Percentage, deserializedPolicy.Rules?[0].Percentage);
        }

        [Fact]
        public void RequestContext_JsonSerialization_RoundtripSucceeds()
        {
            // Arrange
            var originalContext = new RequestContext
            {
                UserId = "user123",
                Segment = "premium",
                Region = "us-west-2",
                RequestTime = DateTime.UtcNow,
                Metadata = new Dictionary<string, string>
                {
                    { "key1", "value1" },
                    { "key2", "value2" }
                }
            };

            // Act
            string json = JsonSerializer.Serialize(originalContext);
            var deserializedContext = JsonSerializer.Deserialize<RequestContext>(json);

            // Assert
            Assert.NotNull(deserializedContext);
            Assert.Equal(originalContext.UserId, deserializedContext.UserId);
            Assert.Equal(originalContext.Segment, deserializedContext.Segment);
            Assert.Equal(originalContext.Region, deserializedContext.Region);
            Assert.Equal(originalContext.Metadata?.Count, deserializedContext.Metadata?.Count);
        }

        [Fact]
        public void RoutingMode_Enum_HasAllExpectedValues()
        {
            // Arrange & Act & Assert
            Assert.Equal(4, Enum.GetValues<RoutingMode>().Length);
            Assert.True(Enum.IsDefined(typeof(RoutingMode), RoutingMode.Percentage));
            Assert.True(Enum.IsDefined(typeof(RoutingMode), RoutingMode.Shadow));
            Assert.True(Enum.IsDefined(typeof(RoutingMode), RoutingMode.Deterministic));
            Assert.True(Enum.IsDefined(typeof(RoutingMode), RoutingMode.TimeBased));
        }

        [Fact]
        public void PercentageValidation_SumEquals100_ValidatesCorrectly()
        {
            // Arrange & Act
            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.Percentage,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0.0", Percentage = 70 },
                    new RoutingRule { Version = "v2.0.0", Percentage = 20 },
                    new RoutingRule { Version = "v3.0.0", Percentage = 10 }
                }
            };

            double sum = policy.Rules!.Sum(r => r.Percentage);

            // Assert
            Assert.Equal(100.0, sum);
        }

        [Fact]
        public void ShadowModeRuleConfiguration_PrimaryVersionSet_Succeeds()
        {
            // Arrange & Act
            var policy = new RoutingPolicy
            {
                Mode = RoutingMode.Shadow,
                Rules = new List<RoutingRule>
                {
                    new RoutingRule { Version = "v1.0.0", IsPrimary = true },
                    new RoutingRule { Version = "v2.0.0", IsPrimary = false }
                }
            };

            // Assert
            Assert.True(policy.Rules![0].IsPrimary);
            Assert.False(policy.Rules[1].IsPrimary);
        }

        [Fact]
        public void TimeBasedRuleConfiguration_TimeRangeWithMultipleDays_Succeeds()
        {
            // Arrange & Act
            var rule = new RoutingRule
            {
                Version = "v1.0.0",
                TimeRange = new TimeRange
                {
                    StartTime = new TimeSpan(9, 0, 0),
                    EndTime = new TimeSpan(17, 0, 0),
                    DaysOfWeek = new[] { DayOfWeek.Monday, DayOfWeek.Wednesday, DayOfWeek.Friday }
                }
            };

            // Assert
            Assert.NotNull(rule.TimeRange);
            Assert.Equal(3, rule.TimeRange.DaysOfWeek.Length);
            Assert.Contains(DayOfWeek.Wednesday, rule.TimeRange.DaysOfWeek);
            Assert.True(rule.TimeRange.IsValid());
        }
    }
}
