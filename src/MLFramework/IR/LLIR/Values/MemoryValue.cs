namespace MLFramework.IR.LLIR.Values
{
    using MLFramework.IR.Types;

    /// <summary>
    /// Represents a memory value in the Low-Level IR (LLIR).
    /// Memory values represent buffer locations with specific offsets and sizes.
    /// </summary>
    public class MemoryValue : LLIRValue
    {
        /// <summary>Gets the memory offset of this buffer in bytes.</summary>
        public int MemoryOffset { get; }

        /// <summary>Gets the size of this buffer in bytes.</summary>
        public int SizeInBytes { get; }

        /// <summary>
        /// Initializes a new instance of the MemoryValue class.
        /// </summary>
        /// <param name="type">The type of this memory value (usually a PointerType).</param>
        /// <param name="name">The name of this memory value.</param>
        /// <param name="offset">The memory offset in bytes.</param>
        /// <param name="sizeInBytes">The size of the buffer in bytes.</param>
        public MemoryValue(IIRType type, string name, int offset, int sizeInBytes)
            : base(type, name, isRegister: false)
        {
            MemoryOffset = offset;
            SizeInBytes = sizeInBytes;
        }

        public override string ToString()
        {
            return $"{Name} : {Type} [offset={MemoryOffset}, size={SizeInBytes}]";
        }
    }
}
