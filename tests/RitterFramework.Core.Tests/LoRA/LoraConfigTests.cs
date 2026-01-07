using System;
using Xunit;
using RitterFramework.Core.LoRA;

namespace RitterFramework.Core.Tests.LoRA
{
    /// <summary>
    /// Unit tests for LoraConfig class
    /// </summary>
    public class LoraConfigTests
    {
        [Fact]
        public void DefaultConstructor_CreatesValidConfig()
        {
            var config = new LoraConfig();

            Assert.Equal(8, config.Rank);
            Assert.Equal(16, config.Alpha);
            Assert.Equal(0.0f, config.Dropout);
            Assert.Equal(new[] { "q_proj", "v_proj" }, config.TargetModules);
            Assert.Equal("none", config.Bias);
            Assert.Equal("default", config.LoraType);
        }

        [Fact]
        public void ConstructorWithParameters_SetsValuesCorrectly()
        {
            var config = new LoraConfig(
                rank: 16,
                alpha: 32,
                dropout: 0.1f,
                targetModules: new[] { "q_proj", "k_proj" },
                bias: "lora_only",
                loraType: "scaled"
            );

            Assert.Equal(16, config.Rank);
            Assert.Equal(32, config.Alpha);
            Assert.Equal(0.1f, config.Dropout);
            Assert.Equal(new[] { "q_proj", "k_proj" }, config.TargetModules);
            Assert.Equal("lora_only", config.Bias);
            Assert.Equal("scaled", config.LoraType);
        }

        [Fact]
        public void Rank_AcceptsPositive()
        {
            var config = new LoraConfig();
            config.Rank = 4;

            Assert.Equal(4, config.Rank);
        }

        [Fact]
        public void Alpha_AcceptsPositive()
        {
            var config = new LoraConfig();
            config.Alpha = 32;

            Assert.Equal(32, config.Alpha);
        }

        [Fact]
        public void Dropout_AcceptsZeroToOne()
        {
            var config = new LoraConfig();

            config.Dropout = 0.0f;
            Assert.Equal(0.0f, config.Dropout);

            config.Dropout = 0.5f;
            Assert.Equal(0.5f, config.Dropout);

            config.Dropout = 1.0f;
            Assert.Equal(1.0f, config.Dropout);
        }

        [Fact]
        public void Bias_AcceptsValidValues()
        {
            var config = new LoraConfig();

            config.Bias = "none";
            Assert.Equal("none", config.Bias);

            config.Bias = "all";
            Assert.Equal("all", config.Bias);

            config.Bias = "lora_only";
            Assert.Equal("lora_only", config.Bias);
        }

        [Fact]
        public void LoraType_AcceptsValidValues()
        {
            var config = new LoraConfig();

            config.LoraType = "default";
            Assert.Equal("default", config.LoraType);

            config.LoraType = "scaled";
            Assert.Equal("scaled", config.LoraType);
        }

        [Fact]
        public void ForLLaMA_CreatesCorrectDefaults()
        {
            var config = LoraConfig.ForLLaMA();

            Assert.Equal(8, config.Rank);
            Assert.Equal(16, config.Alpha);
            Assert.Equal(0.05f, config.Dropout);
            Assert.Contains("q_proj", config.TargetModules);
            Assert.Contains("v_proj", config.TargetModules);
            Assert.Contains("k_proj", config.TargetModules);
            Assert.Contains("o_proj", config.TargetModules);
            Assert.Contains("gate_proj", config.TargetModules);
            Assert.Contains("up_proj", config.TargetModules);
            Assert.Contains("down_proj", config.TargetModules);
            Assert.Equal("none", config.Bias);
            Assert.Equal("default", config.LoraType);
        }

        [Fact]
        public void ForGPT_CreatesCorrectDefaults()
        {
            var config = LoraConfig.ForGPT();

            Assert.Equal(8, config.Rank);
            Assert.Equal(16, config.Alpha);
            Assert.Equal(0.1f, config.Dropout);
            Assert.Contains("c_attn", config.TargetModules);
            Assert.Contains("c_fc", config.TargetModules);
            Assert.Contains("c_proj", config.TargetModules);
            Assert.Equal("none", config.Bias);
            Assert.Equal("default", config.LoraType);
        }

        [Fact]
        public void ForBERT_CreatesCorrectDefaults()
        {
            var config = LoraConfig.ForBERT();

            Assert.Equal(8, config.Rank);
            Assert.Equal(16, config.Alpha);
            Assert.Equal(0.1f, config.Dropout);
            Assert.Contains("query", config.TargetModules);
            Assert.Contains("value", config.TargetModules);
            Assert.Contains("key", config.TargetModules);
            Assert.Contains("output", config.TargetModules);
            Assert.Equal("none", config.Bias);
            Assert.Equal("default", config.LoraType);
        }

        [Fact]
        public void Clone_CreatesIndependentCopy()
        {
            var original = LoraConfig.ForLLaMA();
            var copy = original.Clone();

            Assert.Equal(original.Rank, copy.Rank);
            Assert.Equal(original.Alpha, copy.Alpha);
            Assert.Equal(original.Dropout, copy.Dropout);
            Assert.Equal(original.Bias, copy.Bias);
            Assert.Equal(original.LoraType, copy.LoraType);
            Assert.Equal(original.TargetModules, copy.TargetModules);

            // Modify copy and verify original is unchanged
            copy.Rank = 32;
            copy.TargetModules[0] = "modified";

            Assert.Equal(8, original.Rank);
            Assert.Equal("q_proj", original.TargetModules[0]);
            Assert.Equal(32, copy.Rank);
            Assert.Equal("modified", copy.TargetModules[0]);
        }

        [Fact]
        public void Validate_WithValidConfig_DoesNotThrow()
        {
            var config = LoraConfig.ForLLaMA();

            // Should not throw
            config.Validate();
        }

        [Fact]
        public void Validate_WithInvalidRank_Throws()
        {
            var config = LoraConfig.ForLLaMA();
            config.Rank = -1;

            Assert.Throws<InvalidOperationException>(() => config.Validate());
        }

        [Fact]
        public void Validate_WithInvalidAlpha_Throws()
        {
            var config = LoraConfig.ForLLaMA();
            config.Alpha = 0;

            Assert.Throws<InvalidOperationException>(() => config.Validate());
        }

        [Fact]
        public void Validate_WithInvalidDropout_Throws()
        {
            var config = LoraConfig.ForLLaMA();
            config.Dropout = 1.5f;

            Assert.Throws<InvalidOperationException>(() => config.Validate());
        }

        [Fact]
        public void Validate_WithEmptyTargetModules_Throws()
        {
            var config = new LoraConfig();
            config.TargetModules = new string[0];

            Assert.Throws<InvalidOperationException>(() => config.Validate());
        }

        [Fact]
        public void Validate_WithNullTargetModules_Throws()
        {
            var config = new LoraConfig();
            config.TargetModules = null;

            Assert.Throws<InvalidOperationException>(() => config.Validate());
        }

        [Fact]
        public void Validate_WithInvalidBias_Throws()
        {
            var config = LoraConfig.ForLLaMA();
            config.Bias = "invalid";

            Assert.Throws<InvalidOperationException>(() => config.Validate());
        }

        [Fact]
        public void Validate_WithInvalidLoraType_Throws()
        {
            var config = LoraConfig.ForLLaMA();
            config.LoraType = "invalid";

            Assert.Throws<InvalidOperationException>(() => config.Validate());
        }
    }

    /// <summary>
    /// Unit tests for ModuleTargetPattern class
    /// </summary>
    public class ModuleTargetPatternTests
    {
        [Fact]
        public void ExactNameConstructor_CreatesPattern()
        {
            var pattern = new ModuleTargetPattern("q_proj");

            Assert.Equal("q_proj", pattern.ExactName);
            Assert.Null(pattern.Pattern);
        }

        [Fact]
        public void ExactNameConstructor_ThrowsOnNull()
        {
            Assert.Throws<ArgumentException>(() => new ModuleTargetPattern((string)null));
        }

        [Fact]
        public void ExactNameConstructor_ThrowsOnEmpty()
        {
            Assert.Throws<ArgumentException>(() => new ModuleTargetPattern(""));
        }

        [Fact]
        public void RegexConstructor_CreatesPattern()
        {
            var regex = new System.Text.RegularExpressions.Regex("^.*_proj$");
            var pattern = new ModuleTargetPattern(regex);

            Assert.NotNull(pattern.Pattern);
            Assert.Null(pattern.ExactName);
        }

        [Fact]
        public void RegexConstructor_ThrowsOnNull()
        {
            Assert.Throws<ArgumentNullException>(() => new ModuleTargetPattern((System.Text.RegularExpressions.Regex)null));
        }

        [Fact]
        public void StringRegexConstructor_CreatesPattern()
        {
            var pattern = new ModuleTargetPattern("^.*_proj$", useRegex: true);

            Assert.NotNull(pattern.Pattern);
            Assert.Null(pattern.ExactName);
        }

        [Fact]
        public void StringRegexConstructor_ThrowsOnNull()
        {
            Assert.Throws<ArgumentException>(() => new ModuleTargetPattern(null, useRegex: true));
        }

        [Fact]
        public void StringRegexConstructor_ThrowsOnEmpty()
        {
            Assert.Throws<ArgumentException>(() => new ModuleTargetPattern("", useRegex: true));
        }

        [Fact]
        public void Matches_ExactName_MatchesCorrectly()
        {
            var pattern = new ModuleTargetPattern("q_proj");

            Assert.True(pattern.Matches("q_proj"));
            Assert.False(pattern.Matches("v_proj"));
            Assert.False(pattern.Matches("q_proj_extra"));
        }

        [Fact]
        public void Matches_Regex_MatchesCorrectly()
        {
            var pattern = new ModuleTargetPattern(new System.Text.RegularExpressions.Regex(".*_proj$"));

            Assert.True(pattern.Matches("q_proj"));
            Assert.True(pattern.Matches("v_proj"));
            Assert.True(pattern.Matches("k_proj"));
            Assert.False(pattern.Matches("q_proj_extra"));
            Assert.False(pattern.Matches("proj"));
        }

        [Fact]
        public void Matches_NullModuleName_ReturnsFalse()
        {
            var pattern = new ModuleTargetPattern("q_proj");

            Assert.False(pattern.Matches(null));
        }

        [Fact]
        public void Matches_EmptyModuleName_ReturnsFalse()
        {
            var pattern = new ModuleTargetPattern("q_proj");

            Assert.False(pattern.Matches(""));
        }

        [Fact]
        public void FromString_ExactName_CreatesExactMatch()
        {
            var pattern = ModuleTargetPattern.FromString("q_proj");

            Assert.Equal("q_proj", pattern.ExactName);
            Assert.True(pattern.Matches("q_proj"));
            Assert.False(pattern.Matches("v_proj"));
        }

        [Fact]
        public void FromString_WithWildcard_CreatesRegex()
        {
            var pattern = ModuleTargetPattern.FromString("*_proj");

            Assert.NotNull(pattern.Pattern);
            Assert.True(pattern.Matches("q_proj"));
            Assert.True(pattern.Matches("v_proj"));
            Assert.False(pattern.Matches("proj"));
        }

        [Fact]
        public void FromString_WithQuestionMark_CreatesRegex()
        {
            var pattern = ModuleTargetPattern.FromString("q?proj");

            Assert.NotNull(pattern.Pattern);
            Assert.True(pattern.Matches("q_proj"));
            Assert.True(pattern.Matches("qAproj"));
            Assert.False(pattern.Matches("q__proj"));
        }

        [Fact]
        public void FromString_WithBrackets_CreatesRegex()
        {
            var pattern = ModuleTargetPattern.FromString("q[kv]_proj");

            Assert.NotNull(pattern.Pattern);
            Assert.True(pattern.Matches("qk_proj"));
            Assert.True(pattern.Matches("qv_proj"));
            Assert.False(pattern.Matches("qo_proj"));
        }

        [Fact]
        public void FromString_ThrowsOnNull()
        {
            Assert.Throws<ArgumentException>(() => ModuleTargetPattern.FromString(null));
        }

        [Fact]
        public void FromString_ThrowsOnEmpty()
        {
            Assert.Throws<ArgumentException>(() => ModuleTargetPattern.FromString(""));
        }
    }
}
