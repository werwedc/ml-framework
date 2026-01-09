using System;
using System.Collections.Generic;

namespace MLFramework.Diagnostics
{
    /// <summary>
    /// Represents the result of a validation operation on tensor shapes.
    /// </summary>
    public class ValidationResult
    {
        /// <summary>
        /// Gets or sets whether the validation passed.
        /// </summary>
        public bool IsValid { get; set; }

        /// <summary>
        /// Gets or sets a list of error messages encountered during validation.
        /// </summary>
        public List<string> Errors { get; set; }

        /// <summary>
        /// Gets or sets a list of warning messages encountered during validation.
        /// </summary>
        public List<string> Warnings { get; set; }

        /// <summary>
        /// Creates a successful validation result.
        /// </summary>
        /// <returns>A ValidationResult with IsValid set to true.</returns>
        public static ValidationResult Success()
        {
            return new ValidationResult { IsValid = true };
        }

        /// <summary>
        /// Creates a failed validation result with specified error messages.
        /// </summary>
        /// <param name="errors">The error messages describing why validation failed.</param>
        /// <returns>A ValidationResult with IsValid set to false and the specified errors.</returns>
        public static ValidationResult Failure(params string[] errors)
        {
            return new ValidationResult
            {
                IsValid = false,
                Errors = new List<string>(errors)
            };
        }

        /// <summary>
        /// Initializes a new instance of ValidationResult.
        /// </summary>
        public ValidationResult()
        {
            IsValid = true;
            Errors = new List<string>();
            Warnings = new List<string>();
        }

        /// <summary>
        /// Adds an error message to the validation result.
        /// </summary>
        /// <param name="error">The error message to add.</param>
        public void AddError(string error)
        {
            IsValid = false;
            Errors.Add(error);
        }

        /// <summary>
        /// Adds a warning message to the validation result.
        /// </summary>
        /// <param name="warning">The warning message to add.</param>
        public void AddWarning(string warning)
        {
            Warnings.Add(warning);
        }

        /// <summary>
        /// Returns a string representation of the validation result.
        /// </summary>
        /// <returns>A formatted string containing errors and warnings.</returns>
        public override string ToString()
        {
            if (IsValid && (Warnings == null || Warnings.Count == 0))
            {
                return "Validation passed.";
            }

            var parts = new List<string>();
            if (Errors != null && Errors.Count > 0)
            {
                parts.Add($"Errors: {string.Join("; ", Errors)}");
            }
            if (Warnings != null && Warnings.Count > 0)
            {
                parts.Add($"Warnings: {string.Join("; ", Warnings)}");
            }

            return string.Join(" | ", parts);
        }
    }
}
