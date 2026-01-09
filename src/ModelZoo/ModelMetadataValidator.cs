using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text.RegularExpressions;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Validates ModelMetadata objects to ensure they meet all requirements.
    /// </summary>
    public static class ModelMetadataValidator
    {
        private static readonly Regex SemanticVersionRegex = new Regex(
            @"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$",
            RegexOptions.Compiled);

        private static readonly Regex Sha256Regex = new Regex(
            @"^[a-f0-9]{64}$",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        private static readonly Regex UrlRegex = new Regex(
            @"^https?:\/\/.+$",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        private static readonly HashSet<string> AccuracyLikeMetrics = new HashSet<string>
        {
            "accuracy", "top1", "top5", "f1", "precision", "recall", "auc", "map"
        };

        /// <summary>
        /// Validates a ModelMetadata object and returns all validation errors.
        /// </summary>
        /// <param name="metadata">The metadata to validate.</param>
        /// <returns>A list of validation errors. Empty if validation succeeds.</returns>
        public static List<string> Validate(ModelMetadata metadata)
        {
            var errors = new List<string>();

            if (metadata == null)
            {
                errors.Add("Metadata cannot be null");
                return errors;
            }

            // Use DataAnnotations validation
            var validationContext = new ValidationContext(metadata);
            var validationResults = new List<ValidationResult>();
            bool isValid = Validator.TryValidateObject(metadata, validationContext, validationResults, true);

            if (!isValid)
            {
                errors.AddRange(validationResults.Select(r => r.ErrorMessage ?? r.ToString()));
            }

            // Validate semantic version format
            if (!string.IsNullOrEmpty(metadata.Version))
            {
                if (!SemanticVersionRegex.IsMatch(metadata.Version))
                {
                    errors.Add($"Version '{metadata.Version}' does not follow semantic versioning format");
                }
            }

            // Validate URLs
            if (!string.IsNullOrEmpty(metadata.DownloadUrl))
            {
                if (!UrlRegex.IsMatch(metadata.DownloadUrl))
                {
                    errors.Add($"Download URL '{metadata.DownloadUrl}' is not a valid URL");
                }
            }

            if (!string.IsNullOrEmpty(metadata.PaperUrl))
            {
                if (!UrlRegex.IsMatch(metadata.PaperUrl))
                {
                    errors.Add($"Paper URL '{metadata.PaperUrl}' is not a valid URL");
                }
            }

            if (!string.IsNullOrEmpty(metadata.SourceCodeUrl))
            {
                if (!UrlRegex.IsMatch(metadata.SourceCodeUrl))
                {
                    errors.Add($"Source code URL '{metadata.SourceCodeUrl}' is not a valid URL");
                }
            }

            foreach (var mirrorUrl in metadata.MirrorUrls)
            {
                if (!string.IsNullOrEmpty(mirrorUrl) && !UrlRegex.IsMatch(mirrorUrl))
                {
                    errors.Add($"Mirror URL '{mirrorUrl}' is not a valid URL");
                }
            }

            // Validate SHA256 checksum
            if (!string.IsNullOrEmpty(metadata.Sha256Checksum))
            {
                if (!Sha256Regex.IsMatch(metadata.Sha256Checksum))
                {
                    errors.Add($"SHA256 checksum '{metadata.Sha256Checksum}' is not a valid 64-character hex string");
                }
            }

            // Validate performance metrics ranges
            foreach (var metric in metadata.PerformanceMetrics)
            {
                string metricName = metric.Key.ToLowerInvariant();
                double value = metric.Value;

                // Check if this is an accuracy-like metric (should be 0-1)
                if (AccuracyLikeMetrics.Contains(metricName) || metricName.StartsWith("top"))
                {
                    if (value < 0 || value > 1)
                    {
                        errors.Add($"Performance metric '{metric.Key}' has value {value} but should be in range [0, 1]");
                    }
                }

                // Negative values don't make sense for most metrics
                if (value < 0)
                {
                    errors.Add($"Performance metric '{metric.Key}' has negative value {value}");
                }
            }

            // Validate numeric fields
            if (metadata.NumParameters < 0)
            {
                errors.Add("Number of parameters cannot be negative");
            }

            if (metadata.FileSizeBytes < 0)
            {
                errors.Add("File size cannot be negative");
            }

            // Validate shapes
            if (metadata.InputShape.Length == 0)
            {
                errors.Add("Input shape cannot be empty");
            }

            foreach (int dim in metadata.InputShape)
            {
                if (dim <= 0)
                {
                    errors.Add($"Input shape contains invalid dimension: {dim}");
                }
            }

            return errors;
        }

        /// <summary>
        /// Validates a ModelMetadata object and throws an exception if invalid.
        /// </summary>
        /// <param name="metadata">The metadata to validate.</param>
        /// <exception cref="ValidationException">Thrown when validation fails.</exception>
        public static void ValidateOrThrow(ModelMetadata metadata)
        {
            var errors = Validate(metadata);
            if (errors.Any())
            {
                throw new ValidationException($"ModelMetadata validation failed:\n{string.Join("\n", errors)}");
            }
        }

        /// <summary>
        /// Checks if a ModelMetadata object is valid.
        /// </summary>
        /// <param name="metadata">The metadata to check.</param>
        /// <returns>True if valid, false otherwise.</returns>
        public static bool IsValid(ModelMetadata metadata)
        {
            return Validate(metadata).Count == 0;
        }
    }
}
