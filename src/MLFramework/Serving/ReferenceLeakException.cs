using System;

namespace MLFramework.Serving
{
    /// <summary>
    /// Exception thrown when a reference leak is detected or reference management fails.
    /// </summary>
    public class ReferenceLeakException : Exception
    {
        /// <summary>
        /// Gets the name of the model associated with the reference leak.
        /// </summary>
        public string ModelName { get; }

        /// <summary>
        /// Gets the version of the model associated with the reference leak.
        /// </summary>
        public string Version { get; }

        /// <summary>
        /// Initializes a new instance of the ReferenceLeakException class.
        /// </summary>
        public ReferenceLeakException()
            : base("A reference leak was detected.")
        {
            ModelName = null;
            Version = null;
        }

        /// <summary>
        /// Initializes a new instance of the ReferenceLeakException class with a specified message.
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        public ReferenceLeakException(string message)
            : base(message)
        {
            ModelName = null;
            Version = null;
        }

        /// <summary>
        /// Initializes a new instance of the ReferenceLeakException class with model information.
        /// </summary>
        /// <param name="modelName">The name of the model.</param>
        /// <param name="version">The version of the model.</param>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        public ReferenceLeakException(string modelName, string version, string message)
            : base(message)
        {
            ModelName = modelName;
            Version = version;
        }

        /// <summary>
        /// Initializes a new instance of the ReferenceLeakException class with serialized data.
        /// </summary>
        /// <param name="info">The System.Runtime.Serialization.SerializationInfo that holds the serialized object data.</param>
        /// <param name="context">The System.Runtime.Serialization.StreamingContext that contains contextual information about the source or destination.</param>
        protected ReferenceLeakException(
            System.Runtime.Serialization.SerializationInfo info,
            System.Runtime.Serialization.StreamingContext context)
            : base(info, context)
        {
            ModelName = info.GetString(nameof(ModelName));
            Version = info.GetString(nameof(Version));
        }
    }
}
