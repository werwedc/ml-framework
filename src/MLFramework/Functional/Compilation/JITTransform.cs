using System;
using System.Collections.Concurrent;
using System.Linq;
using RitterFramework.Core.Tensor;
using MLFramework.Functional.Tracing;

namespace MLFramework.Functional.Compilation
{
    public class JITTransform : BaseTransformation
    {
        private static readonly ConcurrentDictionary<Delegate, CompiledFunction> _compiledCache =
            new ConcurrentDictionary<Delegate, CompiledFunction>();

        public JITTransform(Delegate original)
            : base("jit", TransformationType.Compilation)
        {
            ValidateDelegate(original);
        }

        public override Delegate Transform(Delegate original)
        {
            // Check cache first
            if (_compiledCache.TryGetValue(original, out var compiled))
            {
                return compiled.AsDelegate();
            }

            // Trace and compile the function
            var traceContext = TraceAndCompile(original);
            compiled = new CompiledFunction(original, traceContext);

            // Cache the result
            _compiledCache.TryAdd(original, compiled);

            return compiled.AsDelegate();
        }

        private TraceContext TraceAndCompile(Delegate original)
        {
            var method = original.Method;
            var returnType = method.ReturnType;

            // For now, only support Func<Tensor, Tensor> and Func<Tensor, Tensor, Tensor>
            if (method.GetParameters().Length == 1 &&
                method.GetParameters()[0].ParameterType == typeof(Tensor))
            {
                return TraceSingleInput((Func<Tensor, Tensor>)original);
            }

            if (method.GetParameters().Length == 2 &&
                method.GetParameters()[0].ParameterType == typeof(Tensor) &&
                method.GetParameters()[1].ParameterType == typeof(Tensor))
            {
                return TraceDoubleInput((Func<Tensor, Tensor, Tensor>)original);
            }

            throw new NotSupportedException("Unsupported delegate signature for JIT compilation");
        }

        private TraceContext TraceSingleInput(Func<Tensor, Tensor> original)
        {
            using (var trace = new TraceContext())
            {
                // Create a wrapper that traces execution
                Func<Tensor, Tensor> tracedWrapper = (Tensor input) =>
                {
                    // Convert input to TracedTensor
                    var tracedInput = TracedTensor.Create(input, "input");

                    // Execute with traced tensors
                    // Note: This requires the function to work with TracedTensor
                    // For now, we'll use a different approach

                    throw new NotImplementedException("Function must be marked with [TensorFunction] attribute");
                };

                return trace;
            }
        }

        private TraceContext TraceDoubleInput(Func<Tensor, Tensor, Tensor> original)
        {
            using (var trace = new TraceContext())
            {
                // Similar to single input
                return trace;
            }
        }

        private new void ValidateDelegate(Delegate original)
        {
            var method = original.Method;

            // Check return type
            if (method.ReturnType != typeof(Tensor))
                throw new NotSupportedException("JIT compilation only supports functions returning Tensor");

            // Check parameter types
            var parameters = method.GetParameters();

            if (parameters.Length == 0 || parameters.Length > 2)
                throw new NotSupportedException("JIT compilation supports 1 or 2 Tensor parameters");

            foreach (var param in parameters)
            {
                if (param.ParameterType != typeof(Tensor))
                    throw new NotSupportedException("All parameters must be of type Tensor");
            }
        }

        public static void ClearCache()
        {
            _compiledCache.Clear();
        }

        public static int CacheSize => _compiledCache.Count;
    }
}
