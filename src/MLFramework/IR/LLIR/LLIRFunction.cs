using System.Collections.Generic;

namespace MLFramework.IR.LLIR
{
    using MLFramework.IR.Backend;
    using MLFramework.IR.LLIR.Operations;
    using MLFramework.IR.LLIR.Values;
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;

    /// <summary>
    /// Low-Level IR (LLIR) function.
    /// Represents a function at the low-level IR level with explicit memory operations,
    /// register allocation, and low-level control flow.
    /// </summary>
    public class LLIRFunction
    {
        /// <summary>Gets or sets the name of this function.</summary>
        public string Name { get; set; }

        /// <summary>Gets the IR context this function belongs to.</summary>
        public IRContext Context { get; }

        /// <summary>Gets whether this function is a kernel (e.g., GPU kernel).</summary>
        public bool IsKernel { get; }

        /// <summary>Gets the list of register values allocated in this function.</summary>
        public List<LLIRValue> Registers { get; }

        /// <summary>Gets the memory layout for this function.</summary>
        public MemoryLayout MemoryLayout { get; }

        /// <summary>Gets the list of basic blocks in this function.</summary>
        public List<IRBlock> Blocks { get; }

        /// <summary>Gets the function parameters.</summary>
        public List<LLIRValue> Parameters { get; }

        /// <summary>Gets the return value (null for void functions).</summary>
        public LLIRValue ReturnValue { get; set; }

        private int _registerCounter;
        private int _bufferCounter;

        /// <summary>
        /// Initializes a new instance of the LLIRFunction class.
        /// </summary>
        /// <param name="name">The name of the function.</param>
        /// <param name="context">The IR context this function belongs to.</param>
        /// <param name="isKernel">Whether this function is a kernel.</param>
        public LLIRFunction(string name, IRContext context, bool isKernel = false)
        {
            Name = name ?? throw new System.ArgumentNullException(nameof(name));
            Context = context ?? throw new System.ArgumentNullException(nameof(context));
            IsKernel = isKernel;

            Registers = new List<LLIRValue>();
            Blocks = new List<IRBlock>();
            Parameters = new List<LLIRValue>();

            // Default memory layout
            MemoryLayout = Backend.MemoryLayout.RowMajor;

            _registerCounter = 0;
            _bufferCounter = 0;
        }

        /// <summary>
        /// Allocates a new register value for the given type.
        /// </summary>
        /// <param name="type">The type of the register.</param>
        /// <param name="name">Optional name for the register.</param>
        /// <returns>The allocated register value.</returns>
        public LLIRValue AllocateRegister(IIRType type, string name = null)
        {
            var registerName = name ?? $"r{_registerCounter++}";
            var register = new RegisterValue(type, registerName);
            Registers.Add(register);
            return register;
        }

        /// <summary>
        /// Allocates a new buffer in memory.
        /// </summary>
        /// <param name="sizeInBytes">The size of the buffer in bytes.</param>
        /// <param name="alignment">The alignment requirement in bytes.</param>
        /// <param name="name">Optional name for the buffer.</param>
        /// <returns The allocated buffer value.</returns>
        public LLIRValue AllocateBuffer(int sizeInBytes, int alignment = 16, string name = null)
        {
            if (sizeInBytes <= 0)
            {
                throw new System.ArgumentOutOfRangeException(nameof(sizeInBytes), "Buffer size must be positive.");
            }

            var bufferName = name ?? $"buf{_bufferCounter++}";
            var pointerType = new PointerType(new ScalarType(DataType.Float32));
            var buffer = new MemoryValue(pointerType, bufferName, 0, sizeInBytes);
            Registers.Add(buffer);
            return buffer;
        }

        /// <summary>
        /// Adds a basic block to this function.
        /// </summary>
        /// <param name="block">The block to add.</param>
        public void AddBlock(IRBlock block)
        {
            if (block == null)
            {
                throw new System.ArgumentNullException(nameof(block));
            }

            Blocks.Add(block);
        }

        /// <summary>
        /// Adds a parameter to this function.
        /// </summary>
        /// <param name="parameter">The parameter value to add.</param>
        public void AddParameter(LLIRValue parameter)
        {
            if (parameter == null)
            {
                throw new System.ArgumentNullException(nameof(parameter));
            }

            Parameters.Add(parameter);
        }

        /// <summary>
        /// Gets the entry block of this function (first block).
        /// </summary>
        /// <returns>The entry block, or null if no blocks exist.</returns>
        public IRBlock GetEntryBlock()
        {
            return Blocks.Count > 0 ? Blocks[0] : null;
        }
    }
}
