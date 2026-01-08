using System;
using System.Collections.Generic;

namespace MLFramework.Shapes
{
    /// <summary>
    /// Provides a fluent API for building a list of shape constraints.
    /// </summary>
    public sealed class ShapeConstraintBuilder
    {
        private readonly List<IShapeConstraint> _constraints;

        /// <summary>
        /// Initializes a new instance of the ShapeConstraintBuilder class.
        /// </summary>
        public ShapeConstraintBuilder()
        {
            _constraints = new List<IShapeConstraint>();
        }

        /// <summary>
        /// Adds a minimum value constraint to the builder.
        /// </summary>
        /// <param name="value">The minimum value (inclusive).</param>
        /// <returns>This builder instance for method chaining.</returns>
        /// <exception cref="ArgumentException">Thrown when value is negative.</exception>
        public ShapeConstraintBuilder Min(int value)
        {
            if (value < 0)
                throw new ArgumentException("Min value must be non-negative.", nameof(value));

            _constraints.Add(new RangeConstraint(value, int.MaxValue));
            return this;
        }

        /// <summary>
        /// Adds a maximum value constraint to the builder.
        /// </summary>
        /// <param name="value">The maximum value (inclusive).</param>
        /// <returns>This builder instance for method chaining.</returns>
        /// <exception cref="ArgumentException">Thrown when value is negative.</exception>
        public ShapeConstraintBuilder Max(int value)
        {
            if (value < 0)
                throw new ArgumentException("Max value must be non-negative.", nameof(value));

            _constraints.Add(new RangeConstraint(0, value));
            return this;
        }

        /// <summary>
        /// Adds a range constraint with both minimum and maximum values to the builder.
        /// </summary>
        /// <param name="min">The minimum value (inclusive).</param>
        /// <param name="max">The maximum value (inclusive).</param>
        /// <returns>This builder instance for method chaining.</returns>
        /// <exception cref="ArgumentException">Thrown when min > max or values are negative.</exception>
        public ShapeConstraintBuilder Range(int min, int max)
        {
            _constraints.Add(new RangeConstraint(min, max));
            return this;
        }

        /// <summary>
        /// Adds an equality constraint to the builder.
        /// </summary>
        /// <param name="value">The target value that the dimension must equal.</param>
        /// <returns>This builder instance for method chaining.</returns>
        /// <exception cref="ArgumentException">Thrown when value is negative.</exception>
        public ShapeConstraintBuilder Equal(int value)
        {
            _constraints.Add(new EqualityConstraint(value));
            return this;
        }

        /// <summary>
        /// Adds a modulo constraint to the builder.
        /// </summary>
        /// <param name="divisor">The divisor that the dimension value must be divisible by.</param>
        /// <returns>This builder instance for method chaining.</returns>
        /// <exception cref="ArgumentException">Thrown when divisor is less than or equal to zero.</exception>
        public ShapeConstraintBuilder Modulo(int divisor)
        {
            _constraints.Add(new ModuloConstraint(divisor));
            return this;
        }

        /// <summary>
        /// Adds a custom constraint to the builder.
        /// </summary>
        /// <param name="constraint">The custom constraint to add.</param>
        /// <returns>This builder instance for method chaining.</returns>
        /// <exception cref="ArgumentNullException">Thrown when constraint is null.</exception>
        public ShapeConstraintBuilder AddConstraint(IShapeConstraint constraint)
        {
            if (constraint == null)
                throw new ArgumentNullException(nameof(constraint));

            _constraints.Add(constraint);
            return this;
        }

        /// <summary>
        /// Builds and returns the list of constraints.
        /// </summary>
        /// <returns>A list containing all the constraints added to this builder.</returns>
        public List<IShapeConstraint> Build()
        {
            return new List<IShapeConstraint>(_constraints);
        }

        /// <summary>
        /// Resets the builder, removing all previously added constraints.
        /// </summary>
        /// <returns>This builder instance for method chaining.</returns>
        public ShapeConstraintBuilder Reset()
        {
            _constraints.Clear();
            return this;
        }
    }
}
