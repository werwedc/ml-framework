using System;

namespace MLFramework.IR.Transformations
{
    using MLFramework.IR.Graph;

    /// <summary>
    /// Base class for all IR transformations (optimizations, lowerings, analyses, etc.)
    /// </summary>
    public abstract class IRTransformation
    {
        /// <summary>
        /// Gets the name of this transformation
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gets whether this transformation is analysis-only (doesn't modify the IR)
        /// </summary>
        public bool IsAnalysisOnly { get; }

        /// <summary>
        /// Initializes the transformation with the given module
        /// </summary>
        /// <param name="module">The module to initialize with</param>
        public virtual void Initialize(HLIRModule module)
        {
            // Default: no initialization needed
        }

        /// <summary>
        /// Runs the transformation on the given module
        /// </summary>
        /// <param name="module">The module to transform</param>
        /// <returns>True if the transformation modified the module, false otherwise</returns>
        public abstract bool Run(HLIRModule module);

        /// <summary>
        /// Cleans up resources after running the transformation
        /// </summary>
        public virtual void Cleanup()
        {
            // Default: no cleanup needed
        }

        /// <summary>
        /// Initializes a new IRTransformation instance
        /// </summary>
        /// <param name="name">The name of the transformation</param>
        /// <param name="isAnalysisOnly">Whether this transformation is analysis-only</param>
        protected IRTransformation(string name, bool isAnalysisOnly = false)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            IsAnalysisOnly = isAnalysisOnly;
        }
    }
}
