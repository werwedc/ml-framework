using System;

namespace MLFramework.Amp
{
    /// <summary>
    /// Default precision rules for common operations
    /// Note: This uses placeholder types for operations that will be implemented elsewhere
    /// </summary>
    public static class DefaultAmpRules
    {
        // Placeholder types for operations (these will be defined in other modules)
        private class Conv2d { }
        private class Conv3d { }
        private class Linear { }
        private class MaxPool2d { }
        private class MaxPool3d { }
        private class AvgPool2d { }
        private class AvgPool3d { }
        private class BatchNorm1d { }
        private class BatchNorm2d { }
        private class BatchNorm3d { }
        private class Dropout { }
        private class Relu { }
        private class Gelu { }
        private class Sigmoid { }
        private class Tanh { }
        private class ElementwiseAdd { }
        private class ElementwiseMul { }
        private class ElementwiseSub { }
        private class ElementwiseDiv { }
        private class MatMul { }
        private class MatrixMultiply { }
        private class Softmax { }
        private class LogSoftmax { }
        private class Log { }
        private class Exp { }
        private class Sqrt { }
        private class ReduceSum { }
        private class ReduceMean { }
        private class ReduceMax { }
        private class ReduceMin { }
        private class Normalize { }
        private class LayerNorm { }
        private class Embedding { }
        private class CrossEntropyLoss { }
        private class NllLoss { }
        private class KlDivLoss { }
        private class PoissonNLLLoss { }

        /// <summary>
        /// Gets the default whitelist (operations safe for FP16/BF16)
        /// </summary>
        public static Type[] DefaultWhitelist => new Type[]
        {
            typeof(Conv2d),
            typeof(Conv3d),
            typeof(Linear),
            typeof(MaxPool2d),
            typeof(MaxPool3d),
            typeof(AvgPool2d),
            typeof(AvgPool3d),
            typeof(BatchNorm1d),
            typeof(BatchNorm2d),
            typeof(BatchNorm3d),
            typeof(Dropout),
            typeof(Relu),
            typeof(Gelu),
            typeof(Sigmoid),
            typeof(Tanh),
            typeof(ElementwiseAdd),
            typeof(ElementwiseMul),
            typeof(ElementwiseSub),
            typeof(ElementwiseDiv),
            typeof(MatMul),
            typeof(MatrixMultiply)
        };

        /// <summary>
        /// Gets the default blacklist (operations requiring FP32)
        /// </summary>
        public static Type[] DefaultBlacklist => new Type[]
        {
            typeof(Softmax),
            typeof(LogSoftmax),
            typeof(Log),
            typeof(Exp),
            typeof(Sqrt),
            typeof(ReduceSum),
            typeof(ReduceMean),
            typeof(ReduceMax),
            typeof(ReduceMin),
            typeof(Normalize),
            typeof(LayerNorm),
            typeof(Embedding),
            typeof(CrossEntropyLoss),
            typeof(NllLoss),
            typeof(KlDivLoss),
            typeof(PoissonNLLLoss)
        };

        /// <summary>
        /// Applies default rules to the registry
        /// </summary>
        public static void ApplyDefaultRules(AmpRegistry registry)
        {
            if (registry == null)
                throw new ArgumentNullException(nameof(registry));

            // Register whitelist operations
            foreach (var opType in DefaultWhitelist)
            {
                registry.RegisterWhitelist(opType, priority: 0);
            }

            // Register blacklist operations
            foreach (var opType in DefaultBlacklist)
            {
                registry.RegisterBlacklist(opType, priority: 0);
            }
        }
    }
}
