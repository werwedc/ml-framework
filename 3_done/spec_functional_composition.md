# Spec: Function Composition Utilities

## Overview
Implement utility functions for composing and manipulating functions, enabling building complex computations from simple, pure functions.

## Scope
- Implement Compose method for function composition
- Implement Pipe method for left-to-right composition
- Implement Curry and Partial application
- Support for chaining transformations

## Technical Requirements

### 1. Compose Method

```csharp
namespace MLFramework.Functional
{
    public static class Functional
    {
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
    }
}
```

### 2. Pipe Method (Left-to-Right Composition)

```csharp
public static class Functional
{
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
}
```

### 3. Partial Application

```csharp
public static class Functional
{
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
}
```

### 4. Currying

```csharp
public static class Functional
{
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
}
```

### 5. Identity and Constant Functions

```csharp
public static class Functional
{
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
}
```

### 6. Tap (Side-Effect) Function

```csharp
public static class Functional
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
```

## Files to Create
1. `src/MLFramework/Functional/Composition.cs` (or update Functional.cs)
2. Add extension methods to appropriate files

## Dependencies
- spec_functional_core_interfaces.md
- MLFramework.Tensor

## Success Criteria
- Compose works right-to-left
- Pipe works left-to-right
- Partial application reduces function arity
- Currying transforms multi-arg functions
- Identity and constant functions work
- Tap executes side effects while passing through value

## Notes for Coder
- These are pure utilities - no ML framework integration needed
- Focus on functional programming patterns
- Make sure to handle generic types correctly
- Pipe method is an extension method on T
- Include comprehensive XML documentation
- These utilities are useful independently of other functional transformations

## Example Usage
```csharp
// Basic composition
Func<Tensor, Tensor> preprocess = t => t.Normalize();
Func<Tensor, Tensor> forward = t => Model(t);
Func<Tensor, Tensor> postprocess = t => t.Softmax();

var pipeline = Functional.Compose(postprocess, forward, preprocess);
// pipeline(x) = postprocess(forward(preprocess(x)))

// Pipe (left-to-right)
var result = input
    .Pipe(preprocess)
    .Pipe(forward)
    .Pipe(postprocess);

// Partial application
Func<Tensor, Tensor> addOne = Functional.Partial((Tensor a, Tensor b) => a + b, Tensor.Ones());
var result = addOne(input);  // input + ones

// Currying
var curriedAdd = Functional.Curry((int a, int b) => a + b);
var addFive = curriedAdd(5);
var result = addFive(3);  // 8

// Tap for debugging
var result = input
    .Pipe(preprocess)
    .Tap(x => Console.WriteLine($"After preprocess: {x.Shape}"))
    .Pipe(forward);
```
