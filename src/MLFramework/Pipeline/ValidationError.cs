using System;
using System.Collections.Generic;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Represents a validation error
    /// </summary>
    public class ValidationError
    {
        /// <summary>
        /// Error code
        /// </summary>
        public string Code { get; }

        /// <summary>
        /// Error message
        /// </summary>
        public string Message { get; }

        /// <summary>
        /// Stage index where error occurred (-1 if not stage-specific)
        /// </summary>
        public int StageIndex { get; }

        /// <summary>
        /// Severity (Error = must fix)
        /// </summary>
        public ValidationSeverity Severity { get; }

        /// <summary>
        /// Additional context
        /// </summary>
        public Dictionary<string, object> Context { get; }

        public ValidationError(
            string code,
            string message,
            int stageIndex = -1,
            Dictionary<string, object>? context = null)
            : this(code, message, stageIndex, ValidationSeverity.Error, context)
        {
        }

        protected ValidationError(
            string code,
            string message,
            int stageIndex,
            ValidationSeverity severity,
            Dictionary<string, object>? context)
        {
            Code = code ?? throw new ArgumentNullException(nameof(code));
            Message = message ?? throw new ArgumentNullException(nameof(message));
            StageIndex = stageIndex;
            Severity = severity;
            Context = context ?? new Dictionary<string, object>();
        }

        public override string ToString()
        {
            var stageInfo = StageIndex >= 0 ? $" [Stage {StageIndex}]" : "";
            return $"[{Severity}] {Code}{stageInfo}: {Message}";
        }
    }

    /// <summary>
    /// Severity levels for validation issues
    /// </summary>
    public enum ValidationSeverity
    {
        /// <summary>
        /// Must fix before proceeding
        /// </summary>
        Error,

        /// <summary>
        /// Should fix but can proceed
        /// </summary>
        Warning,

        /// <summary>
        /// Informational only
        /// </summary>
        Info
    }

    /// <summary>
    /// Represents a validation warning
    /// </summary>
    public class ValidationWarning : ValidationError
    {
        public ValidationWarning(
            string code,
            string message,
            int stageIndex = -1,
            Dictionary<string, object>? context = null)
            : base(code, message, stageIndex, ValidationSeverity.Warning, context)
        {
        }
    }
}
