namespace MLFramework.Shapes
{
    /// <summary>
    /// Factory class for creating SymbolicDimension instances with common configurations.
    /// </summary>
    public static class SymbolicDimensionFactory
    {
        /// <summary>
        /// Creates a symbolic dimension with the specified name and optional value.
        /// </summary>
        /// <param name="name">The name of the dimension (e.g., "batch_size", "seq_len").</param>
        /// <param name="value">The concrete value if known, or null for unknown.</param>
        /// <returns>A new SymbolicDimension instance.</returns>
        /// <exception cref="System.ArgumentNullException">Thrown when name is null or whitespace.</exception>
        public static SymbolicDimension Create(string name, int? value = null)
        {
            return new SymbolicDimension(name, value);
        }

        /// <summary>
        /// Creates a symbolic dimension with both minimum and maximum bounds.
        /// </summary>
        /// <param name="name">The name of the dimension.</param>
        /// <param name="min">The minimum value (lower bound).</param>
        /// <param name="max">The maximum value (upper bound).</param>
        /// <returns>A new SymbolicDimension instance with bounded range.</returns>
        /// <exception cref="System.ArgumentNullException">Thrown when name is null or whitespace.</exception>
        /// <exception cref="System.ArgumentException">Thrown when bounds are invalid.</exception>
        public static SymbolicDimension CreateBounded(string name, int min, int max)
        {
            return new SymbolicDimension(name, null, min, max);
        }

        /// <summary>
        /// Creates a symbolic dimension with a minimum bound but unbounded maximum.
        /// </summary>
        /// <param name="name">The name of the dimension.</param>
        /// <param name="min">The minimum value (lower bound).</param>
        /// <returns>A new SymbolicDimension instance with unbounded maximum.</returns>
        /// <exception cref="System.ArgumentNullException">Thrown when name is null or whitespace.</exception>
        /// <exception cref="System.ArgumentException">Thrown when min is negative.</exception>
        public static SymbolicDimension CreateRange(string name, int min)
        {
            return new SymbolicDimension(name, null, min, null);
        }

        /// <summary>
        /// Creates a symbolic dimension with a concrete known value.
        /// </summary>
        /// <param name="name">The name of the dimension.</param>
        /// <param name="value">The concrete value.</param>
        /// <returns>A new SymbolicDimension instance with a known value.</returns>
        /// <exception cref="System.ArgumentNullException">Thrown when name is null or whitespace.</exception>
        public static SymbolicDimension CreateKnown(string name, int value)
        {
            return new SymbolicDimension(name, value);
        }

        /// <summary>
        /// Creates a symbolic dimension representing a batch size.
        /// Typically used with no value but with a minimum bound of 1.
        /// </summary>
        /// <returns>A new SymbolicDimension for batch size.</returns>
        public static SymbolicDimension BatchSize()
        {
            return new SymbolicDimension("batch_size", null, 1, null);
        }

        /// <summary>
        /// Creates a symbolic dimension representing a sequence length.
        /// Typically used with no value but with a minimum bound of 1.
        /// </summary>
        /// <returns>A new SymbolicDimension for sequence length.</returns>
        public static SymbolicDimension SequenceLength()
        {
            return new SymbolicDimension("seq_len", null, 1, null);
        }

        /// <summary>
        /// Creates a symbolic dimension representing a feature dimension.
        /// Usually has a known concrete value.
        /// </summary>
        /// <param name="value">The number of features.</param>
        /// <returns>A new SymbolicDimension for features.</returns>
        public static SymbolicDimension Features(int value)
        {
            return new SymbolicDimension("features", value);
        }

        /// <summary>
        /// Creates a symbolic dimension representing channels in a convolutional layer.
        /// Usually has a known concrete value.
        /// </summary>
        /// <param name="value">The number of channels.</param>
        /// <returns>A new SymbolicDimension for channels.</returns>
        public static SymbolicDimension Channels(int value)
        {
            return new SymbolicDimension("channels", value);
        }
    }
}
