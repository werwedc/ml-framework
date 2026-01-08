using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLFramework.Cache
{
    /// <summary>
    /// Represents a unique signature for an operation based on operation name and input shapes.
    /// Used for indexing cached compiled kernels.
    /// </summary>
    public struct ShapeSignature : IEquatable<ShapeSignature>
    {
        /// <summary>
        /// Gets the name of the operation (e.g., "MatMul", "Conv2D").
        /// </summary>
        public string OperationName { get; }

        /// <summary>
        /// Gets the list of concrete input shapes (one per input tensor).
        /// Each shape is an array of dimension sizes.
        /// </summary>
        public int[][] InputShapes { get; }

        /// <summary>
        /// Gets the precomputed hash for fast lookup.
        /// </summary>
        public int Hash { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ShapeSignature"/> struct.
        /// </summary>
        /// <param name="operationName">Name of the operation.</param>
        /// <param name="inputShapes">List of concrete input shapes.</param>
        /// <param name="hash">Precomputed hash value.</param>
        private ShapeSignature(string operationName, int[][] inputShapes, int hash)
        {
            OperationName = operationName;
            InputShapes = inputShapes;
            Hash = hash;
        }

        /// <summary>
        /// Creates a new <see cref="ShapeSignature"/> from operation name and input shapes.
        /// </summary>
        /// <param name="operationName">Name of the operation.</param>
        /// <param name="shapes">List of concrete input shapes.</param>
        /// <returns>A new <see cref="ShapeSignature"/> instance.</returns>
        public static ShapeSignature Create(string operationName, List<int[]> shapes)
        {
            if (string.IsNullOrEmpty(operationName))
            {
                throw new ArgumentException("Operation name cannot be null or empty.", nameof(operationName));
            }

            if (shapes == null)
            {
                throw new ArgumentNullException(nameof(shapes));
            }

            // Deep copy shapes to ensure immutability
            var shapesCopy = new int[shapes.Count][];
            for (int i = 0; i < shapes.Count; i++)
            {
                if (shapes[i] == null)
                {
                    throw new ArgumentException($"Shape at index {i} cannot be null.", nameof(shapes));
                }
                shapesCopy[i] = (int[])shapes[i].Clone();
            }

            int hash = ComputeHash(operationName, shapesCopy);
            return new ShapeSignature(operationName, shapesCopy, hash);
        }

        /// <summary>
        /// Computes a hash value for the operation name and input shapes.
        /// </summary>
        private static int ComputeHash(string operationName, int[][] shapes)
        {
            var hash = new HashCode();

            // Include operation name in hash
            hash.Add(operationName);

            // Include all shape dimensions in hash
            foreach (var shape in shapes)
            {
                foreach (int dim in shape)
                {
                    hash.Add(dim);
                }
            }

            return hash.ToHashCode();
        }

        /// <summary>
        /// Determines whether the specified <see cref="ShapeSignature"/> is equal to the current instance.
        /// </summary>
        /// <param name="other">The <see cref="ShapeSignature"/> to compare.</param>
        /// <returns>True if equal; otherwise, false.</returns>
        public bool Equals(ShapeSignature other)
        {
            if (!string.Equals(OperationName, other.OperationName, StringComparison.Ordinal))
            {
                return false;
            }

            if (InputShapes.Length != other.InputShapes.Length)
            {
                return false;
            }

            for (int i = 0; i < InputShapes.Length; i++)
            {
                var thisShape = InputShapes[i];
                var otherShape = other.InputShapes[i];

                if (thisShape.Length != otherShape.Length)
                {
                    return false;
                }

                for (int j = 0; j < thisShape.Length; j++)
                {
                    if (thisShape[j] != otherShape[j])
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current instance.
        /// </summary>
        /// <param name="obj">The object to compare.</param>
        /// <returns>True if equal; otherwise, false.</returns>
        public override bool Equals(object? obj)
        {
            return obj is ShapeSignature other && Equals(other);
        }

        /// <summary>
        /// Returns the hash code for this instance.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode()
        {
            return Hash;
        }

        /// <summary>
        /// Returns a string representation of this signature.
        /// </summary>
        /// <returns>A string describing the operation and input shapes.</returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append(OperationName);
            sb.Append('(');

            for (int i = 0; i < InputShapes.Length; i++)
            {
                if (i > 0)
                {
                    sb.Append(", ");
                }

                sb.Append('[');
                for (int j = 0; j < InputShapes[i].Length; j++)
                {
                    if (j > 0)
                    {
                        sb.Append(", ");
                    }
                    sb.Append(InputShapes[i][j]);
                }
                sb.Append(']');
            }

            sb.Append(')');
            return sb.ToString();
        }

        /// <summary>
        /// Determines whether two <see cref="ShapeSignature"/> instances are equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if equal; otherwise, false.</returns>
        public static bool operator ==(ShapeSignature left, ShapeSignature right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Determines whether two <see cref="ShapeSignature"/> instances are not equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if not equal; otherwise, false.</returns>
        public static bool operator !=(ShapeSignature left, ShapeSignature right)
        {
            return !left.Equals(right);
        }
    }
}
