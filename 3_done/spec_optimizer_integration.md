# Spec: Higher-Order Derivatives Optimizer Integration

## Overview
Integrate higher-order derivative computations with existing optimizer classes. Enable second-order optimization methods (Newton's method, natural gradient) and enhance first-order optimizers with curvature information.

## Requirements

### Core Functionality
- Extend existing optimizer interfaces to support Hessian/HVP information
- Implement second-order optimizers (Newton's method, quasi-Newton)
- Add natural gradient optimization support
- Enable Hessian-aware learning rate adaptation

### API Design

#### Optimizer Base Extensions
```csharp
// Extend Optimizer base class
public abstract class SecondOrderOptimizer : Optimizer
{
    // Hessian computation options
    public HessianMode HessianMode { get; set; } = HessianMode.HVP;
    public bool UseCheckpointing { get; set; } = false;

    // Abstract method to compute update step
    protected abstract Tensor ComputeUpdateStep(
        Model model,
        Tensor loss,
        Tensor[] gradients,
        Tensor[] parameters);

    // Hessian computation helpers
    protected Tensor ComputeHVP(Tensor loss, Tensor[] parameters, Tensor vector);
    protected Tensor ComputeHessian(Tensor loss, Tensor[] parameters);
}
```

#### Newton's Method Optimizer
```csharp
public class NewtonOptimizer : SecondOrderOptimizer
{
    public NewtonOptimizer(double learningRate = 1.0, double damping = 1e-4)
    {
        LearningRate = learningRate;
        Damping = damping; // Add damping to avoid singular Hessian
    }

    public double LearningRate { get; set; }
    public double Damping { get; set; }

    protected override Tensor ComputeUpdateStep(
        Model model,
        Tensor loss,
        Tensor[] gradients,
        Tensor[] parameters)
    {
        // Solve: (H + λI) * Δp = -g
        // Use Hessian-vector products with conjugate gradient
        var hessianInverseGradient = SolveLinearSystem(loss, parameters, gradients);
        return -LearningRate * hessianInverseGradient;
    }

    private Tensor SolveLinearSystem(Tensor loss, Tensor[] parameters, Tensor[] gradients)
    {
        // Solve (H + λI) * x = b using conjugate gradient
        // Use HVP for matrix-free multiplication
        return ConjugateGradient(
            b: gradients,
            matrixVectorProduct: v => ComputeHVP(loss, parameters, v) + Damping * v);
    }
}
```

#### Quasi-Newton Optimizer (BFGS/L-BFGS)
```csharp
public class LBFGSOptimizer : Optimizer
{
    public int HistorySize { get; set; } = 10;
    public int MaxIterations { get; set; } = 20;
    public double Tolerance { get; set; } = 1e-5;

    // L-BFGS state
    private List<Tensor> sHistory; // Parameter differences
    private List<Tensor> yHistory; // Gradient differences

    protected override Tensor[] ComputeGradients(Model model, Tensor loss, Tensor[] parameters)
    {
        var gradients = base.ComputeGradients(model, loss, parameters);

        // Compute update step using L-BFGS two-loop recursion
        var updateStep = ComputeLBFGSUpdate(gradients);

        // Update history
        UpdateHistory(gradients);

        return updateStep;
    }

    private Tensor ComputeLBFGSUpdate(Tensor[] gradients)
    {
        // Implement two-loop recursion algorithm
        // Uses gradient history to approximate Hessian inverse
    }
}
```

#### Natural Gradient Optimizer
```csharp
public class NaturalGradientOptimizer : SecondOrderOptimizer
{
    public NaturalGradientOptimizer(double learningRate = 1e-3)
    {
        LearningRate = learningRate;
    }

    public double LearningRate { get; set; }

    protected override Tensor ComputeUpdateStep(
        Model model,
        Tensor loss,
        Tensor[] gradients,
        Tensor[] parameters)
    {
        // Natural gradient: F^(-1) * ∇L
        // Where F is Fisher information matrix (approximated as Hessian)
        var fisherInverseGradient = SolveFisherSystem(loss, parameters, gradients);
        return -LearningRate * fisherInverseGradient;
    }

    private Tensor SolveFisherSystem(Tensor loss, Tensor[] parameters, Tensor[] gradients)
    {
        // Solve F * x = g using HVP (F ≈ Hessian for expected loss)
        return ConjugateGradient(
            b: gradients,
            matrixVectorProduct: v => ComputeHVP(loss, parameters, v));
    }
}
```

#### Enhanced First-Order Optimizers
```csharp
// Adam with adaptive learning rate based on Hessian eigenvalues
public class HessianAwareAdam : Adam
{
    protected override Tensor[] ComputeGradients(Model model, Tensor loss, Tensor[] parameters)
    {
        var gradients = base.ComputeGradients(model, loss, parameters);

        // Estimate maximum eigenvalue of Hessian
        var maxEigenvalue = EstimateMaxEigenvalue(loss, parameters);

        // Adapt learning rate based on curvature
        var adaptedLearningRate = LearningRate / (1 + maxEigenvalue);

        return AdaptGradients(gradients, adaptedLearningRate);
    }

    private double EstimateMaxEigenvalue(Tensor loss, Tensor[] parameters)
    {
        // Use power iteration with HVP
        return PowerIteration(
            matrixVectorProduct: v => ComputeHVP(loss, parameters, v),
            numIterations: 10);
    }
}
```

### Helper Classes

#### Conjugate Gradient Solver
```csharp
public static class ConjugateGradient
{
    public static Tensor Solve(
        Tensor b,
        Func<Tensor, Tensor> matrixVectorProduct,
        int maxIterations = 100,
        double tolerance = 1e-10)
    {
        // Solve Ax = b using conjugate gradient
        // Uses matrix-vector product (HVP) without materializing A
    }
}
```

#### Power Iteration for Eigenvalues
```csharp
public static class PowerIteration
{
    public static double EstimateMaxEigenvalue(
        Func<Tensor, Tensor> matrixVectorProduct,
        int dimension,
        int numIterations = 20)
    {
        // Estimate maximum eigenvalue using power iteration
    }
}
```

## Implementation Tasks

### Phase 1: Optimizer Base Extensions
1. Extend Optimizer base class with Hessian computation helpers
2. Implement SecondOrderOptimizer abstract class
3. Add HessianMode and configuration options
4. Add base unit tests

### Phase 2: Newton's Method
1. Implement NewtonOptimizer class
2. Implement conjugate gradient solver
3. Add damping for numerical stability
4. Add Newton optimizer unit tests

### Phase 3: Quasi-Newton Methods
1. Implement LBFGSOptimizer class
3. Implement two-loop recursion algorithm
4. Add history management
5. Add L-BFGS unit tests

### Phase 4: Natural Gradient
1. Implement NaturalGradientOptimizer class
2. Add Fisher information matrix approximation
3. Add natural gradient unit tests

### Phase 5: Enhanced First-Order Optimizers
1. Implement HessianAwareAdam class
2. Implement eigenvalue estimation
3. Add adaptive learning rate logic
4. Add enhanced optimizer unit tests

## Testing Requirements

### Optimizer Correctness Tests
- Test Newton's method converges on quadratic functions
- Test L-BFGS converges on non-convex optimization problems
- Test natural gradient improves convergence for specific problems
- Test Hessian-aware Adam adapts learning rate correctly

### Convergence Tests
- Test Newton's method converges faster than SGD on convex problems
- Test L-BFGS memory efficiency vs full BFGS
- Test natural gradient on probabilistic models
- Test enhanced Adam on problems with varying curvature

### Stability Tests
- Test Newton's method with singular Hessian (damping)
- Test L-BFGS with ill-conditioned problems
- Test natural gradient with approximated Fisher
- Test Hessian-aware Adam numerical stability

### Performance Tests
- Benchmark Newton's method vs first-order optimizers
- Benchmark L-BFGS vs BFGS memory usage
- Benchmark natural gradient vs standard gradient descent
- Benchmark Hessian-aware Adam vs standard Adam

## Dependencies
- HVP implementation (spec_hvp_implementation.md)
- Hessian implementation (spec_hessian_full.md)
- Existing optimizer classes
- Linear algebra utilities

## Success Criteria
- Newton's method converges in < 10 iterations on quadratic problems
- L-BFGS uses < 10% memory of full BFGS for large problems
- Natural gradient improves convergence on probabilistic models
- Hessian-aware Adam adapts learning rate based on curvature
- All optimizers are stable and numerically robust

## Notes for Coder
- Focus on numerical stability - second-order methods can be unstable
- Conjugate gradient is critical - implement carefully with proper stopping criteria
- L-BFGS is the most practical for large-scale problems - prioritize this
- Natural gradient is experimentally useful - focus on approximations
- Consider implementing trust region methods as alternative to line search
- Add extensive logging for debugging convergence issues
- Test with various loss landscapes (convex, non-convex, ill-conditioned)
- Document when to use each optimizer and their trade-offs
