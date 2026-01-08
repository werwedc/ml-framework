using System;
using System.Linq;
using RitterFramework.Core.Tensor;

namespace MLFramework.Functional
{
    /// <summary>
    /// Provides utility functions for composing and manipulating functions,
    /// enabling building complex computations from simple, pure functions.
    /// </summary>
    public static class Functional
    {
        #region Compose Methods

        /// <summary>
        /// Compose functions from right to left: Compose(f, g)(x) = f(g(x)).
        /// </summary>
        /// <param name="outer">Outer function</param>
        /// <param name="inner">Inner function</param>
        /// <returns>Composed function</returns>
        public static Func<T, TRes> Compose<T, TIntermediate, TRes>(
            Func<TIntermediate, TRes> outer,
            Func<T, TIntermediate> inner)
        {
            return x => outer(inner(x));
        }

        /// <summary>
        /// Compose three functions: Compose(f, g, h)(x) = f(g(h(x))).
        /// </summary>
        public static Func<T, TRes> Compose<T, T1, T2, TRes>(
            Func<T2, TRes> f,
            Func<T1, T2> g,
            Func<T, T1> h)
        {
            return Compose(Compose(f, g), h);
        }

        /// <summary>
        /// Compose an arbitrary number of functions (right-to-left).
        /// </summary>
        public static Func<T, T> Compose<T>(params Func<T, T>[] functions)
        {
            if (functions == null || functions.Length == 0)
                throw new ArgumentException("At least one function required");

            if (functions.Length == 1)
                return functions[0];

            return functions.Aggregate((f, g) => Compose(f, g));
        }

        /// <summary>
        /// Compose functions for tensor operations.
        /// </summary>
        public static Func<Tensor, Tensor> Compose(
            params Func<Tensor, Tensor>[] functions)
        {
            return Compose<Tensor>(functions);
        }

        #endregion

        #region Pipe Methods

        /// <summary>
        /// Pipe value through functions left-to-right: Pipe(x, f, g) = g(f(x)).
        /// </summary>
        public static T Pipe<T>(this T value, Func<T, T> func)
        {
            return func(value);
        }

        /// <summary>
        /// Pipe value through multiple functions left-to-right.
        /// </summary>
        public static T Pipe<T>(this T value, params Func<T, T>[] functions)
        {
            return functions.Aggregate(value, (acc, func) => func(acc));
        }

        /// <summary>
        /// Pipe value through functions with different types.
        /// </summary>
        public static TRes Pipe<T, TRes>(this T value, params Delegate[] functions)
        {
            object current = value;

            foreach (var func in functions)
            {
                current = func.DynamicInvoke(current);
            }

            return (TRes)current;
        }

        #endregion

        #region Partial Application

        /// <summary>
        /// Partially apply a function with the first argument.
        /// </summary>
        public static Func<T2, TRes> Partial<T1, T2, TRes>(
            Func<T1, T2, TRes> func,
            T1 arg1)
        {
            return (arg2) => func(arg1, arg2);
        }

        /// <summary>
        /// Partially apply a function with the second argument.
        /// </summary>
        public static Func<T1, TRes> PartialSecond<T1, T2, TRes>(
            Func<T1, T2, TRes> func,
            T2 arg2)
        {
            return (arg1) => func(arg1, arg2);
        }

        /// <summary>
        /// Partially apply a 3-argument function.
        /// </summary>
        public static Func<T2, T3, TRes> Partial<T1, T2, T3, TRes>(
            Func<T1, T2, T3, TRes> func,
            T1 arg1)
        {
            return (arg2, arg3) => func(arg1, arg2, arg3);
        }

        #endregion

        #region Currying

        /// <summary>
        /// Curry a 2-argument function: Curry(f)(a)(b) = f(a, b).
        /// </summary>
        public static Func<T1, Func<T2, TRes>> Curry<T1, T2, TRes>(
            Func<T1, T2, TRes> func)
        {
            return a => b => func(a, b);
        }

        /// <summary>
        /// Curry a 3-argument function: Curry(f)(a)(b)(c) = f(a, b, c).
        /// </summary>
        public static Func<T1, Func<T2, Func<T3, TRes>>> Curry<T1, T2, T3, TRes>(
            Func<T1, T2, T3, TRes> func)
        {
            return a => b => c => func(a, b, c);
        }

        /// <summary>
        /// Uncurry a curried function.
        /// </summary>
        public static Func<T1, T2, TRes> Uncurry<T1, T2, TRes>(
            Func<T1, Func<T2, TRes>> curried)
        {
            return (a, b) => curried(a)(b);
        }

        #endregion

        #region Identity and Constant Functions

        /// <summary>
        /// Identity function: Identity(x) = x.
        /// </summary>
        public static T Identity<T>(T value) => value;

        /// <summary>
        /// Constant function: Constant(c)(x) = c.
        /// </summary>
        public static Func<T, T> Constant<T>(T value)
        {
            return _ => value;
        }

        #endregion

        #region Vectorization

        /// <summary>
        /// Transforms a function that operates on single tensors to work on batches.
        /// </summary>
        /// <param name="func">Function to vectorize.</param>
        /// <param name="axis">Batch axis (default: 0).</param>
        /// <returns>Batched function.</returns>
        public static Func<Tensor, Tensor> Vectorize(
            Func<Tensor, Tensor> func,
            int axis = 0)
        {
            var transform = new VMapTransform(func, axis);
            return (Func<Tensor, Tensor>)transform.Transform(func);
        }

        /// <summary>
        /// Transforms a function that operates on single tensors to work on batches with per-parameter axis specification.
        /// </summary>
        /// <param name="func">Function to vectorize.</param>
        /// <param name="in_axes">Array of axes for each parameter. Use null for non-batched parameters.</param>
        /// <returns>Batched function.</returns>
        public static Func<Tensor, Tensor> Vectorize(
            Func<Tensor, Tensor> func,
            object[] in_axes)
        {
            var transform = new VMapTransform(func, in_axes);
            return (Func<Tensor, Tensor>)transform.Transform(func);
        }

        /// <summary>
        /// Vectorize a function with multiple input tensors.
        /// </summary>
        /// <param name="func">Function to vectorize with two inputs.</param>
        /// <param name="axis">Batch axis (default: 0).</param>
        /// <returns>Batched function.</returns>
        public static Func<Tensor, Tensor, Tensor> Vectorize(
            Func<Tensor, Tensor, Tensor> func,
            int axis = 0)
        {
            var transform = new VMapTransform(func, axis);
            return (Func<Tensor, Tensor, Tensor>)transform.Transform(func);
        }

        /// <summary>
        /// Vectorize a function with multiple input tensors with per-parameter axis specification.
        /// </summary>
        /// <param name="func">Function to vectorize with two inputs.</param>
        /// <param name="in_axes">Array of axes for each parameter. Use null for non-batched parameters.</param>
        /// <returns>Batched function.</returns>
        public static Func<Tensor, Tensor, Tensor> Vectorize(
            Func<Tensor, Tensor, Tensor> func,
            object[] in_axes)
        {
            var transform = new VMapTransform(func, in_axes);
            return (Func<Tensor, Tensor, Tensor>)transform.Transform(func);
        }

        #endregion

        #region Parallelization

        /// <summary>
        /// Parallelizes a function across a device mesh (SPMD).
        /// </summary>
        /// <param name="func">Function to parallelize.</param>
        /// <param name="mesh">Device mesh for distribution.</param>
        /// <param name="in_axes">Which axes to shard across devices (default: data axis).</param>
        /// <returns>Parallel function that returns sharded result.</returns>
        public static Func<Tensor, Tensor> Parallelize(
            Func<Tensor, Tensor> func,
            Distributed.DeviceMesh mesh,
            object[] in_axes = null)
        {
            var transform = new Distributed.PMapTransform(func, mesh, in_axes);
            return (Func<Tensor, Tensor>)transform.Transform(func);
        }

        /// <summary>
        /// Parallelizes a function with multiple inputs.
        /// </summary>
        /// <param name="func">Function to parallelize.</param>
        /// <param name="mesh">Device mesh for distribution.</param>
        /// <param name="in_axes">Which axes to shard across devices (default: data axis).</param>
        /// <returns>Parallel function that returns sharded result.</returns>
        public static Func<Tensor, Tensor, Tensor> Parallelize(
            Func<Tensor, Tensor, Tensor> func,
            Distributed.DeviceMesh mesh,
            object[] in_axes = null)
        {
            var transform = new Distributed.PMapTransform(func, mesh, in_axes);
            return (Func<Tensor, Tensor, Tensor>)transform.Transform(func);
        }

        #endregion

        #region JIT Compilation

        /// <summary>
        /// Just-in-time compile a function for optimization.
        /// </summary>
        public static Func<Tensor, Tensor> Compile(Func<Tensor, Tensor> func)
        {
            var transform = new Compilation.JITTransform(func);
            return (Func<Tensor, Tensor>)transform.Transform(func);
        }

        public static Func<Tensor, Tensor, Tensor> Compile(Func<Tensor, Tensor, Tensor> func)
        {
            var transform = new Compilation.JITTransform(func);
            return (Func<Tensor, Tensor, Tensor>)transform.Transform(func);
        }

        /// <summary>
        /// Clear the JIT compilation cache.
        /// </summary>
        public static void ClearJITCache()
        {
            Compilation.JITTransform.ClearCache();
        }

        #endregion
    }

    #region Tap Extension Methods

    /// <summary>
    /// Extension methods for the Tap (side-effect) functionality.
    /// </summary>
    public static class TapExtensions
    {
        /// <summary>
        /// Execute an action and return the value: Tap(x, f) returns x after calling f(x).
        /// Useful for logging or debugging.
        /// </summary>
        public static T Tap<T>(this T value, Action<T> action)
        {
            action(value);
            return value;
        }

        /// <summary>
        /// Execute an action with condition.
        /// </summary>
        public static T TapIf<T>(this T value, bool condition, Action<T> action)
        {
            if (condition)
                action(value);
            return value;
        }
    }

    #endregion
}
