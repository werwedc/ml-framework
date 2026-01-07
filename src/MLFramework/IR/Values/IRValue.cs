using System;

namespace MLFramework.IR.Values
{
    using MLFramework.IR.Types;

    /// <summary>
    /// Represents a value in the IR system.
    /// Values can be operation results, function parameters, or constants.
    /// </summary>
    public class IRValue
    {
        private static int _nextId = 0;

        /// <summary>Gets the type of this value.</summary>
        public IIRType Type { get; }

        /// <summary>Gets the name of this value.</summary>
        public string Name { get; }

        /// <summary>Gets the unique ID of this value.</summary>
        public int Id { get; }

        /// <summary>
        /// Initializes a new instance of the IRValue class.
        /// </summary>
        /// <param name="type">The type of this value.</param>
        /// <param name="name">The name of this value (optional).</param>
        internal IRValue(IIRType type, string name = null)
        {
            if (type == null)
            {
                throw new ArgumentNullException(nameof(type));
            }

            Type = type;
            Name = name ?? GenerateDefaultName();
            Id = _nextId++;
        }

        /// <summary>
        /// Generates a default name for the value.
        /// </summary>
        /// <returns>A default name in the format "v{Id}".</returns>
        private string GenerateDefaultName()
        {
            return $"v{Id}";
        }

        public override string ToString()
        {
            return $"{Name} : {Type}";
        }

        public override bool Equals(object obj)
        {
            if (obj is not IRValue other)
                return false;

            return Id == other.Id;
        }

        public override int GetHashCode()
        {
            return Id.GetHashCode();
        }
    }
}
