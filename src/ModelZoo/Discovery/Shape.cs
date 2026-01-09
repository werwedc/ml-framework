using System;
using System.Linq;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Represents tensor shape for model input/output.
    /// </summary>
    public class Shape
    {
        /// <summary>
        /// Gets the dimensions of the shape.
        /// </summary>
        public int[] Dimensions { get; }

        /// <summary>
        /// Gets the rank (number of dimensions).
        /// </summary>
        public int Rank => Dimensions.Length;

        /// <summary>
        /// Initializes a new instance of the Shape class.
        /// </summary>
        /// <param name="dimensions">The dimensions of the shape.</param>
        public Shape(int[] dimensions)
        {
            Dimensions = dimensions ?? throw new ArgumentNullException(nameof(dimensions));
        }

        /// <summary>
        /// Checks if this shape matches another shape exactly.
        /// </summary>
        /// <param name="other">The other shape to compare.</param>
        /// <returns>True if shapes match exactly.</returns>
        public bool MatchesExactly(Shape other)
        {
            if (other == null)
                return false;

            if (Rank != other.Rank)
                return false;

            return Dimensions.SequenceEqual(other.Dimensions);
        }

        /// <summary>
        /// Checks if this shape is compatible with another shape (allows variable dimensions like batch size).
        /// </summary>
        /// <param name="other">The other shape to compare.</param>
        /// <returns>True if shapes are compatible.</returns>
        public bool IsCompatibleWith(Shape other)
        {
            if (other == null)
                return false;

            if (Rank != other.Rank)
                return false;

            // Allow flexible dimensions (e.g., -1 for variable batch size)
            for (int i = 0; i < Rank; i++)
            {
                if (Dimensions[i] >= 0 && other.Dimensions[i] >= 0 &&
                    Dimensions[i] != other.Dimensions[i])
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Gets the total number of elements in the shape.
        /// </summary>
        /// <returns>Total number of elements.</returns>
        public long TotalElements()
        {
            long total = 1;
            foreach (int dim in Dimensions)
            {
                if (dim > 0)
                    total *= dim;
            }
            return total;
        }

        /// <summary>
        /// Converts the shape to an integer array.
        /// </summary>
        /// <returns>Array of dimensions.</returns>
        public int[] ToArray()
        {
            return (int[])Dimensions.Clone();
        }

        /// <summary>
        /// Gets a string representation of the shape.
        /// </summary>
        /// <returns>String representation.</returns>
        public override string ToString()
        {
            return $"[{string.Join(", ", Dimensions)}]";
        }

        /// <summary>
        /// Creates a Shape from an integer array.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns>A new Shape instance.</returns>
        public static Shape FromArray(int[] dimensions)
        {
            return new Shape(dimensions);
        }
    }
}
