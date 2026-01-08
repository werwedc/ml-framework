namespace MLFramework.Shapes
{
    /// <summary>
    /// Factory class for creating SymbolicShape instances with various convenience methods.
    /// </summary>
    public static class SymbolicShapeFactory
    {
        /// <summary>
        /// Creates a SymbolicShape from the specified dimensions.
        /// </summary>
        /// <param name="dims">The dimensions of the shape.</param>
        /// <returns>A new SymbolicShape instance.</returns>
        public static SymbolicShape Create(params SymbolicDimension[] dims)
        {
            return new SymbolicShape(dims);
        }

        /// <summary>
        /// Creates a SymbolicShape from concrete integer dimensions.
        /// </summary>
        /// <param name="dims">The concrete dimension values.</param>
        /// <returns>A new SymbolicShape with known dimensions.</returns>
        public static SymbolicShape FromConcrete(params int[] dims)
        {
            if (dims == null)
                throw new ArgumentNullException(nameof(dims));

            var symbolicDims = dims.Select((value, index) =>
                new SymbolicDimension($"dim_{index}", value)).ToArray();
            return new SymbolicShape(symbolicDims);
        }

        /// <summary>
        /// Creates a scalar shape (rank 0).
        /// </summary>
        /// <returns>A scalar SymbolicShape.</returns>
        public static SymbolicShape Scalar()
        {
            return new SymbolicShape(Array.Empty<SymbolicDimension>());
        }

        /// <summary>
        /// Creates a vector shape (rank 1) with a known length.
        /// </summary>
        /// <param name="length">The length of the vector.</param>
        /// <returns>A vector SymbolicShape.</returns>
        public static SymbolicShape Vector(int length)
        {
            return FromConcrete(length);
        }

        /// <summary>
        /// Creates a vector shape (rank 1) with a symbolic length.
        /// </summary>
        /// <param name="name">The name of the dimension.</param>
        /// <returns>A vector SymbolicShape with symbolic length.</returns>
        public static SymbolicShape Vector(string name)
        {
            return new SymbolicShape(new SymbolicDimension(name));
        }

        /// <summary>
        /// Creates a matrix shape (rank 2) with known dimensions.
        /// </summary>
        /// <param name="rows">The number of rows.</param>
        /// <param name="cols">The number of columns.</param>
        /// <returns>A matrix SymbolicShape.</returns>
        public static SymbolicShape Matrix(int rows, int cols)
        {
            return FromConcrete(rows, cols);
        }

        /// <summary>
        /// Creates a matrix shape (rank 2) with symbolic dimensions.
        /// </summary>
        /// <param name="rowsName">The name of the rows dimension.</param>
        /// <param name="colsName">The name of the columns dimension.</param>
        /// <returns>A matrix SymbolicShape with symbolic dimensions.</returns>
        public static SymbolicShape Matrix(string rowsName, string colsName)
        {
            return new SymbolicShape(
                new SymbolicDimension(rowsName),
                new SymbolicDimension(colsName));
        }

        /// <summary>
        /// Creates a batched tensor shape (rank N+1) with a symbolic batch dimension.
        /// </summary>
        /// <param name="batchName">The name of the batch dimension.</param>
        /// <param name="innerShape">The inner shape without batch dimension.</param>
        /// <returns>A batched SymbolicShape.</returns>
        public static SymbolicShape Batched(string batchName, SymbolicShape innerShape)
        {
            if (innerShape == null)
                throw new ArgumentNullException(nameof(innerShape));

            var batchDim = new SymbolicDimension(batchName);
            var allDims = new[] { batchDim }.Concat(innerShape.Dimensions).ToArray();
            return new SymbolicShape(allDims);
        }

        /// <summary>
        /// Creates a batched tensor shape (rank N+1) with a concrete batch size.
        /// </summary>
        /// <param name="batchSize">The size of the batch.</param>
        /// <param name="innerShape">The inner shape without batch dimension.</param>
        /// <returns>A batched SymbolicShape.</returns>
        public static SymbolicShape Batched(int batchSize, SymbolicShape innerShape)
        {
            if (innerShape == null)
                throw new ArgumentNullException(nameof(innerShape));

            var batchDim = new SymbolicDimension("batch", batchSize);
            var allDims = new[] { batchDim }.Concat(innerShape.Dimensions).ToArray();
            return new SymbolicShape(allDims);
        }
    }
}
