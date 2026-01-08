using System;
using System.Collections.Generic;

namespace MLFramework.Functional
{
    /// <summary>
    /// Provides context for transformation execution, including metadata and hierarchical relationships.
    /// </summary>
    public class TransformationContext
    {
        /// <summary>
        /// Gets or sets whether debug mode is enabled for this transformation.
        /// When enabled, additional logging and validation may be performed.
        /// </summary>
        public bool DebugMode { get; set; }

        /// <summary>
        /// Gets the metadata dictionary for this transformation context.
        /// Metadata can include transformation-specific parameters, configuration options, etc.
        /// </summary>
        public Dictionary<string, object> Metadata { get; }

        /// <summary>
        /// Gets the parent transformation context, if this context is nested within another transformation.
        /// Enables hierarchical transformation composition.
        /// </summary>
        public TransformationContext Parent { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="TransformationContext"/> class.
        /// </summary>
        public TransformationContext()
            : this(null)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TransformationContext"/> class with a parent context.
        /// </summary>
        /// <param name="parent">The parent transformation context.</param>
        public TransformationContext(TransformationContext parent)
        {
            Parent = parent;
            Metadata = new Dictionary<string, object>();
            DebugMode = parent?.DebugMode ?? false;
        }

        /// <summary>
        /// Gets a metadata value by key, optionally searching through parent contexts.
        /// </summary>
        /// <typeparam name="T">The type of the metadata value.</typeparam>
        /// <param name="key">The metadata key.</param>
        /// <param name="searchParents">Whether to search parent contexts if not found.</param>
        /// <returns>The metadata value, or default if not found.</returns>
        public T GetMetadata<T>(string key, bool searchParents = true)
        {
            if (Metadata.TryGetValue(key, out var value) && value is T typedValue)
            {
                return typedValue;
            }

            if (searchParents && Parent != null)
            {
                return Parent.GetMetadata<T>(key, true);
            }

            return default;
        }

        /// <summary>
        /// Sets a metadata value.
        /// </summary>
        /// <typeparam name="T">The type of the metadata value.</typeparam>
        /// <param name="key">The metadata key.</param>
        /// <param name="value">The metadata value.</param>
        public void SetMetadata<T>(string key, T value)
        {
            Metadata[key] = value;
        }

        /// <summary>
        /// Creates a child context of this context.
        /// </summary>
        /// <returns>A new transformation context with this context as parent.</returns>
        public TransformationContext CreateChildContext()
        {
            return new TransformationContext(this);
        }
    }
}
