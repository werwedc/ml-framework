using System;

namespace MLFramework.IR.HLIR.ControlFlow
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    public class IfOp : IROperation
    {
        public IRValue Condition { get; }
        private IRValue _result;
        public IRValue Result => _result ?? Results[0];
        public IRBlock TrueBlock { get; }
        public IRBlock FalseBlock { get; }

        public IfOp(IRValue condition, IRValue result, IRBlock trueBlock, IRBlock falseBlock)
            : base("if", IROpcode.IfOp, new[] { condition }, new[] { result.Type }, null)
        {
            Condition = condition ?? throw new ArgumentNullException(nameof(condition));
            _result = result ?? throw new ArgumentNullException(nameof(result));
            TrueBlock = trueBlock ?? throw new ArgumentNullException(nameof(trueBlock));
            FalseBlock = falseBlock ?? throw new ArgumentNullException(nameof(falseBlock));
        }

        public override void Validate()
        {
            if (Condition.Type is not TensorType condType)
                throw new InvalidOperationException("Condition must be a tensor type");
            if (condType.ElementType != DataType.Bool)
                throw new InvalidOperationException("Condition must be boolean type");
            if (TrueBlock.Returns.Count != 1 || FalseBlock.Returns.Count != 1)
                throw new InvalidOperationException("True and false blocks must each return exactly one value");
        }

        public static IRValue Create(IRContext ctx, IRValue condition,
                                   Action<IRBlock> trueBranch, Action<IRBlock> falseBranch,
                                   string name = null)
        {
            if (condition == null)
                throw new ArgumentNullException(nameof(condition));
            if (trueBranch == null)
                throw new ArgumentNullException(nameof(trueBranch));
            if (falseBranch == null)
                throw new ArgumentNullException(nameof(falseBranch));

            var condType = (TensorType)condition.Type;
            var resultType = new TensorType(condType.ElementType, condType.Shape);
            var result = ctx.CreateValue(resultType, name);

            var trueBlock = new IRBlock("if_true");
            var falseBlock = new IRBlock("if_false");

            trueBranch(trueBlock);
            falseBranch(falseBlock);

            var op = new IfOp(condition, result, trueBlock, falseBlock);
            ctx.RegisterOperation(op);
            return result;
        }

        public override IROperation Clone() => new IfOp(Condition, Result, TrueBlock, FalseBlock);
    }

    public class LoopOp : IROperation
    {
        public IRValue InitialValue { get; }
        private IRValue _result;
        public IRValue Result => _result ?? Results[0];
        public IRValue LoopCondition { get; }
        public IRBlock Body { get; }

        public LoopOp(IRValue initialValue, IRValue result, IRValue loopCondition, IRBlock body)
            : base("loop", IROpcode.LoopOp,
                    new[] { initialValue, loopCondition }, new[] { result.Type },
                    null)
        {
            InitialValue = initialValue ?? throw new ArgumentNullException(nameof(initialValue));
            _result = result ?? throw new ArgumentNullException(nameof(result));
            LoopCondition = loopCondition ?? throw new ArgumentNullException(nameof(loopCondition));
            Body = body ?? throw new ArgumentNullException(nameof(body));
        }

        public override void Validate()
        {
            if (InitialValue.Type is not TensorType initType)
                throw new InvalidOperationException("Initial value must be a tensor type");
            if (LoopCondition.Type is not TensorType condType)
                throw new InvalidOperationException("Loop condition must be a tensor type");
            if (condType.ElementType != DataType.Bool)
                throw new InvalidOperationException("Loop condition must be boolean type");
            if (Body.Arguments.Count != 1)
                throw new InvalidOperationException("Loop body must accept exactly one argument");
        }

        public static IRValue Create(IRContext ctx, IRValue initialValue,
                                   Action<IRValue, IRBlock> body,
                                   string name = null)
        {
            if (initialValue == null)
                throw new ArgumentNullException(nameof(initialValue));
            if (body == null)
                throw new ArgumentNullException(nameof(body));

            var initType = (TensorType)initialValue.Type;
            var resultType = new TensorType(initType.ElementType, initType.Shape);
            var result = ctx.CreateValue(resultType, name);

            var loopCondition = ctx.CreateValue(new TensorType(DataType.Bool, new int[] { 1 }), "loop_cond");
            var loopBody = new IRBlock("loop_body");

            body(initialValue, loopBody);

            var op = new LoopOp(initialValue, result, loopCondition, loopBody);
            ctx.RegisterOperation(op);
            return result;
        }

        public override IROperation Clone() => new LoopOp(InitialValue, Result, LoopCondition, Body);
    }
}
