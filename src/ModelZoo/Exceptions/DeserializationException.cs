using System;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Exception thrown when a model file is corrupted or has an invalid format.
    /// </summary>
    public class DeserializationException : Exception
    {
        /// <summary>
        /// Path to the model file that failed deserialization.
        /// </summary>
        public string FilePath { get; }

        /// <summary>
        /// Creates a new DeserializationException.
        /// </summary>
        /// <param name="filePath">Path to the model file that failed deserialization.</param>
        /// <param name="message">Descriptive message about the deserialization failure.</param>
        public DeserializationException(string filePath, string message)
            : base($"Failed to deserialize model file '{filePath}': {message}")
        {
            FilePath = filePath;
        }

        /// <summary>
        /// Creates a new DeserializationException with an inner exception.
        /// </summary>
        /// <param name="filePath">Path to the model file that failed deserialization.</param>
        /// <param name="message">Descriptive message about the deserialization failure.</param>
        /// <param name="innerException">The exception that caused this exception.</param>
        public DeserializationException(string filePath, string message, Exception innerException)
            : base($"Failed to deserialize model file '{filePath}': {message}", innerException)
        {
            FilePath = filePath;
        }
    }
}
