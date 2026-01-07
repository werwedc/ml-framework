using System;
using System.Text.RegularExpressions;

namespace RitterFramework.Core.LoRA
{
    /// <summary>
    /// Configuration class for LoRA (Low-Rank Adaptation) hyperparameters and module targeting.
    /// </summary>
    public class LoraConfig
    {
        /// <summary>Rank of low-rank matrices (default: 8)</summary>
        public int Rank { get; set; } = 8;

        /// <summary>LoRA scaling factor (default: 16)</summary>
        public int Alpha { get; set; } = 16;

        /// <summary>Dropout probability for LoRA layers (default: 0.0)</summary>
        public float Dropout { get; set; } = 0.0f;

        /// <summary>Target module names/patterns to inject LoRA into</summary>
        public string[] TargetModules { get; set; } = new[] { "q_proj", "v_proj" };

        /// <summary>Whether to bias LoRA layers (none, all, lora_only)</summary>
        public string Bias { get; set; } = "none";

        /// <summary>LoRA module type (default, scaled)</summary>
        public string LoraType { get; set; } = "default";

        /// <summary>
        /// Default constructor
        /// </summary>
        public LoraConfig() { }

        /// <summary>
        /// Constructor with custom parameters
        /// </summary>
        public LoraConfig(int rank, int alpha, float dropout, string[] targetModules, string bias = "none", string loraType = "default")
        {
            Rank = rank;
            Alpha = alpha;
            Dropout = dropout;
            TargetModules = targetModules;
            Bias = bias;
            LoraType = loraType;
        }

        /// <summary>
        /// Initialize with default values for LLaMA architecture
        /// </summary>
        public static LoraConfig ForLLaMA()
        {
            return new LoraConfig(
                rank: 8,
                alpha: 16,
                dropout: 0.05f,
                targetModules: new[] { "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj" }
            );
        }

        /// <summary>
        /// Initialize with default values for GPT architecture
        /// </summary>
        public static LoraConfig ForGPT()
        {
            return new LoraConfig(
                rank: 8,
                alpha: 16,
                dropout: 0.1f,
                targetModules: new[] { "c_attn", "c_fc", "c_proj" }
            );
        }

        /// <summary>
        /// Initialize with default values for BERT architecture
        /// </summary>
        public static LoraConfig ForBERT()
        {
            return new LoraConfig(
                rank: 8,
                alpha: 16,
                dropout: 0.1f,
                targetModules: new[] { "query", "value", "key", "output" }
            );
        }

        /// <summary>
        /// Create a copy of this configuration
        /// </summary>
        public LoraConfig Clone()
        {
            return new LoraConfig
            {
                Rank = this.Rank,
                Alpha = this.Alpha,
                Dropout = this.Dropout,
                TargetModules = (string[])this.TargetModules.Clone(),
                Bias = this.Bias,
                LoraType = this.LoraType
            };
        }

        /// <summary>
        /// Validate the configuration
        /// </summary>
        public void Validate()
        {
            if (Rank <= 0)
                throw new InvalidOperationException($"Invalid Rank: {Rank}. Must be greater than 0.");

            if (Alpha <= 0)
                throw new InvalidOperationException($"Invalid Alpha: {Alpha}. Must be greater than 0.");

            if (Dropout < 0.0f || Dropout > 1.0f)
                throw new InvalidOperationException($"Invalid Dropout: {Dropout}. Must be between 0.0 and 1.0.");

            if (TargetModules == null || TargetModules.Length == 0)
                throw new InvalidOperationException("TargetModules cannot be null or empty.");

            if (Bias != "none" && Bias != "all" && Bias != "lora_only")
                throw new InvalidOperationException($"Invalid Bias: {Bias}. Must be 'none', 'all', or 'lora_only'.");

            if (LoraType != "default" && LoraType != "scaled")
                throw new InvalidOperationException($"Invalid LoraType: {LoraType}. Must be 'default' or 'scaled'.");
        }
    }

    /// <summary>
    /// Pattern matching for module names in LoRA injection.
    /// </summary>
    public class ModuleTargetPattern
    {
        /// <summary>Exact module name match</summary>
        public string ExactName { get; set; }

        /// <summary>Regex pattern for module matching</summary>
        public Regex Pattern { get; set; }

        /// <summary>
        /// Constructor for exact name matching
        /// </summary>
        public ModuleTargetPattern(string exactName)
        {
            if (string.IsNullOrEmpty(exactName))
                throw new ArgumentException("Exact name cannot be null or empty", nameof(exactName));

            ExactName = exactName;
            Pattern = null;
        }

        /// <summary>
        /// Constructor for regex pattern matching
        /// </summary>
        public ModuleTargetPattern(Regex pattern)
        {
            Pattern = pattern ?? throw new ArgumentNullException(nameof(pattern));
            ExactName = null;
        }

        /// <summary>
        /// Constructor for regex pattern from string
        /// </summary>
        public ModuleTargetPattern(string patternString, bool useRegex)
        {
            if (useRegex)
            {
                if (string.IsNullOrEmpty(patternString))
                    throw new ArgumentException("Pattern string cannot be null or empty", nameof(patternString));

                Pattern = new Regex(patternString);
                ExactName = null;
            }
            else
            {
                ExactName = patternString;
                Pattern = null;
            }
        }

        /// <summary>
        /// Check if module name matches this pattern
        /// </summary>
        public bool Matches(string moduleName)
        {
            if (string.IsNullOrEmpty(moduleName))
                return false;

            if (ExactName != null)
                return moduleName == ExactName;

            if (Pattern != null)
                return Pattern.IsMatch(moduleName);

            return false;
        }

        /// <summary>
        /// Create a ModuleTargetPattern from a target string
        /// </summary>
        public static ModuleTargetPattern FromString(string target)
        {
            if (string.IsNullOrEmpty(target))
                throw new ArgumentException("Target cannot be null or empty", nameof(target));

            // Check if it's a regex pattern (contains special regex characters)
            if (target.Contains("*") || target.Contains("?") || target.Contains("+") || target.Contains("["))
            {
                // Convert wildcard pattern to regex
                string regexPattern = "^" + Regex.Escape(target).Replace("\\*", ".*").Replace("\\?", ".") + "$";
                return new ModuleTargetPattern(new Regex(regexPattern));
            }
            else
            {
                return new ModuleTargetPattern(target);
            }
        }
    }
}
