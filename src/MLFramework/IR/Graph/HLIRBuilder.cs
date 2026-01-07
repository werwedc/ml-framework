using System;

namespace MLFramework.IR.Graph
{
    using MLFramework.IR.Attributes;
    using MLFramework.IR.HLIR;
    using MLFramework.IR.HLIR.Activation;
    using MLFramework.IR.HLIR.Conv;
    using MLFramework.IR.HLIR.ControlFlow;
    using MLFramework.IR.HLIR.Elementwise;
    using MLFramework.IR.HLIR.Matrix;
    using MLFramework.IR.HLIR.Pool;
    using MLFramework.IR.HLIR.Reduce;
    using MLFramework.IR.HLIR.Shape;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    /// <summary>
    /// Builder class for constructing High-Level IR graphs.
    /// Provides convenient methods for creating IR operations and managing insertion points.
    /// </summary>
    public class HLIRBuilder
    {
        private IRContext _context;
        private HLIRFunction _currentFunction;
        private IRBlock _currentBlock;

        /// <summary>
        /// Initializes a new instance of the HLIRBuilder class with a function.
        /// </summary>
        /// <param name="function">The function to build.</param>
        public HLIRBuilder(HLIRFunction function)
        {
            if (function == null)
            {
                throw new ArgumentNullException(nameof(function));
            }

            _currentFunction = function;
            _context = function.Context;
            _currentBlock = function.Body;
        }

        /// <summary>
        /// Initializes a new instance of the HLIRBuilder class with a module.
        /// Creates a new function and builds into it.
        /// </summary>
        /// <param name="module">The module to build into.</param>
        public HLIRBuilder(HLIRModule module)
        {
            if (module == null)
            {
                throw new ArgumentNullException(nameof(module));
            }

            _context = module.Context;
            _currentFunction = module.CreateFunction("main");
            _currentBlock = _currentFunction.Body;
        }

        /// <summary>
        /// Sets the insertion point to the beginning of a block.
        /// </summary>
        /// <param name="block">The block to insert operations into.</param>
        public void SetInsertPoint(IRBlock block)
        {
            if (block == null)
            {
                throw new ArgumentNullException(nameof(block));
            }

            _currentBlock = block;
        }

        /// <summary>
        /// Sets the insertion point to the end of a block.
        /// </summary>
        /// <param name="block">The block to insert operations into.</param>
        public void SetInsertPointToEnd(IRBlock block)
        {
            SetInsertPoint(block);
        }

        // Tensor operations

        /// <summary>
        /// Creates an element-wise addition operation.
        /// </summary>
        public IRValue Add(IRValue lhs, IRValue rhs, string name = null)
        {
            var result = AddOp.Create(_context, lhs, rhs, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates an element-wise subtraction operation.
        /// </summary>
        public IRValue Sub(IRValue lhs, IRValue rhs, string name = null)
        {
            var result = SubOp.Create(_context, lhs, rhs, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates an element-wise multiplication operation.
        /// </summary>
        public IRValue Mul(IRValue lhs, IRValue rhs, string name = null)
        {
            var result = MulOp.Create(_context, lhs, rhs, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates an element-wise division operation.
        /// </summary>
        public IRValue Div(IRValue lhs, IRValue rhs, string name = null)
        {
            var result = DivOp.Create(_context, lhs, rhs, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        // Activations

        /// <summary>
        /// Creates a ReLU activation operation.
        /// </summary>
        public IRValue ReLU(IRValue input, string name = null)
        {
            var result = ReLUOp.Create(_context, input, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates a sigmoid activation operation.
        /// </summary>
        public IRValue Sigmoid(IRValue input, string name = null)
        {
            var result = SigmoidOp.Create(_context, input, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates a tanh activation operation.
        /// </summary>
        public IRValue Tanh(IRValue input, string name = null)
        {
            var result = TanhOp.Create(_context, input, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates a GELU activation operation.
        /// </summary>
        public IRValue GELU(IRValue input, string name = null)
        {
            var result = GELUOp.Create(_context, input, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        // Matrix ops

        /// <summary>
        /// Creates a matrix multiplication operation.
        /// </summary>
        public IRValue MatMul(IRValue lhs, IRValue rhs,
                              bool transposeA = false, bool transposeB = false,
                              string name = null)
        {
            var result = MatMulOp.Create(_context, lhs, rhs, transposeA, transposeB, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates a 2D convolution operation.
        /// </summary>
        public IRValue Conv2D(IRValue input, IRValue weight, IRValue bias,
                              int[] kernelSize, int[] stride,
                              int[] padding = null, int[] dilation = null,
                              int groups = 1, string name = null)
        {
            var result = Conv2DOp.Create(_context, input, weight, bias, kernelSize, stride,
                                         padding, dilation, groups, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        // Pooling

        /// <summary>
        /// Creates a 2D max pooling operation.
        /// </summary>
        public IRValue MaxPool2D(IRValue input, int[] kernelSize,
                                int[] stride, int[] padding = null,
                                string name = null)
        {
            var result = MaxPool2DOp.Create(_context, input, kernelSize, stride, padding, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates a 2D average pooling operation.
        /// </summary>
        public IRValue AvgPool2D(IRValue input, int[] kernelSize,
                                int[] stride, int[] padding = null,
                                string name = null)
        {
            var result = AvgPool2DOp.Create(_context, input, kernelSize, stride, padding, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        // Reductions

        /// <summary>
        /// Creates a sum reduction operation.
        /// </summary>
        public IRValue ReduceSum(IRValue input, int[] axes,
                                bool keepDims = false, string name = null)
        {
            var result = ReduceSumOp.Create(_context, input, axes, keepDims, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates a mean reduction operation.
        /// </summary>
        public IRValue ReduceMean(IRValue input, int[] axes,
                                 bool keepDims = false, string name = null)
        {
            var result = ReduceMeanOp.Create(_context, input, axes, keepDims, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates a max reduction operation.
        /// </summary>
        public IRValue ReduceMax(IRValue input, int[] axes,
                                bool keepDims = false, string name = null)
        {
            var result = ReduceMaxOp.Create(_context, input, axes, keepDims, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        // Shape ops

        /// <summary>
        /// Creates a reshape operation.
        /// </summary>
        public IRValue Reshape(IRValue input, int[] newShape, string name = null)
        {
            var result = ReshapeOp.Create(_context, input, newShape, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates a transpose operation.
        /// </summary>
        public IRValue Transpose(IRValue input, int[] permutation, string name = null)
        {
            var result = TransposeOp.Create(_context, input, permutation, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates a broadcast operation.
        /// </summary>
        public IRValue BroadcastTo(IRValue input, int[] targetShape, string name = null)
        {
            var result = BroadcastToOp.Create(_context, input, targetShape, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        // Constants

        /// <summary>
        /// Creates a constant operation.
        /// </summary>
        public IRValue Constant(IIRAttribute value, string name = null)
        {
            var result = ConstantOp.Create(_context, value, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        // Control flow

        /// <summary>
        /// Creates a conditional (if) operation.
        /// </summary>
        public IRValue If(IRValue condition, Action<IRBlock> trueBranch,
                          Action<IRBlock> falseBranch, string name = null)
        {
            var result = IfOp.Create(_context, condition, trueBranch, falseBranch, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Creates a loop operation.
        /// </summary>
        public IRValue Loop(IRValue initialValue, Action<IRValue, IRBlock> body,
                            string name = null)
        {
            var result = LoopOp.Create(_context, initialValue, body, name);
            _currentBlock.AddOperation(GetLastOperation());
            return result;
        }

        /// <summary>
        /// Gets the last operation registered in the context.
        /// This is used to add the operation to the current block after creation.
        /// </summary>
        private Operations.IROperation GetLastOperation()
        {
            var ops = _context.GetAllOperations();
            if (ops.Count == 0)
            {
                return null;
            }

            // Get the last registered operation
            // This is a simple approach - in a more sophisticated implementation,
            // we might track the last created operation directly
            var opArray = new Operations.IROperation[ops.Count];
            ops.CopyTo(opArray, 0);
            return opArray[opArray.Length - 1];
        }

        /// <summary>
        /// Gets the current function being built.
        /// </summary>
        public HLIRFunction CurrentFunction => _currentFunction;

        /// <summary>
        /// Gets the current block being built into.
        /// </summary>
        public IRBlock CurrentBlock => _currentBlock;
    }
}
