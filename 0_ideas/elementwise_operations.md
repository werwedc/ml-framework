# Feature: Element-wise Tensor Operations

## Problem
The current tensor implementation only supports basic addition and scalar multiplication. A complete ML framework needs comprehensive element-wise operations for mathematical computations, activation functions, and tensor manipulations.

## Solution
Implement a comprehensive suite of element-wise operations including:
- **Arithmetic operations**: Subtraction (-), Division (/), Element-wise multiplication (*)
- **Mathematical functions**: Exp, Log, Sqrt, Pow, Abs
- **Activation functions**: ReLU, Sigmoid, Tanh, LeakyReLU
- **Comparison operations**: GreaterThan, LessThan, Equal
- **Aggregation operations**: Sum, Mean, Max, Min, ArgMax, ArgMin

## Value
- Enables implementation of complex neural network architectures
- Provides essential building blocks for loss functions and activation layers
- Follows PyTorch/TensorFlow API conventions for familiarity
- All operations support automatic differentiation for training

## Technical Considerations
- Broadcasting support for operations on tensors of different shapes
- Gradient computation for each operation
- In-place operations with proper gradient handling
- Memory efficiency for large tensors

## Priority
**High** - Core functionality needed before any model training can occur

## Dependencies
- Existing Tensor class with gradient tracking
- Broadcasting infrastructure (to be implemented as part of this feature)
