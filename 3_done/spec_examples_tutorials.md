# Spec: Higher-Order Derivatives Examples and Tutorials

## Overview
Create comprehensive examples and tutorials demonstrating the use of higher-order derivatives for common machine learning tasks. Examples should cover MAML, Newton's method optimization, Neural ODEs, and other use cases.

## Requirements

### Example Categories
1. **MAML (Model-Agnostic Meta-Learning)** - Computing gradients of gradients
2. **Newton's Method Optimization** - Using Hessian for faster convergence
3. **Neural ODEs** - Solving differential equations with higher-order derivatives
4. **Sharpness Minimization** - Analyzing loss landscape geometry
5. **Adversarial Robustness** - Computing curvature for attack generation
6. **Natural Gradient** - Using Fisher information matrix

### Example Structure

#### Example 1: MAML Implementation
```csharp
// examples/MAML/MAMLExample.cs
public class MAMLExample
{
    public static void Run()
    {
        // Problem setup: Few-shot learning with inner and outer loops

        // Inner loop: Task-specific adaptation
        Model adaptModel(Task task, Model metaModel, double innerLR, int innerSteps)
        {
            var model = metaModel.Clone();
            for (int step = 0; step < innerSteps; step++)
            {
                var loss = ComputeTaskLoss(model, task);
                var grads = Autograd.Gradient(loss, model.Parameters);
                UpdateParameters(model, grads, innerLR);
            }
            return model;
        }

        // Outer loop: Meta-learning using gradient of gradient
        Model metaModel = InitializeModel();

        foreach (var task in tasks)
        {
            // Inner loop adaptation
            var adaptedModel = adaptModel(task, metaModel, innerLR: 0.01, innerSteps: 5);

            // Outer loop: Compute meta-gradient
            var metaLoss = ComputeTaskLoss(adaptedModel, task);

            // CRITICAL: Compute gradient of gradient using nested differentiation
            using (var outerTape = GradientTape.Record())
            {
                using (var innerTape = GradientTape.Record())
                {
                    var adaptedModel = adaptModelWithTape(task, metaModel, innerTape);
                    var metaLoss = ComputeTaskLoss(adaptedModel, task);
                    var innerGrads = innerTape.Gradient(metaLoss, adaptedModel.Parameters);
                }
                // innerGrads is differentiable!
                var metaGrads = outerTape.Gradient(innerGrads, metaModel.Parameters);
                UpdateParameters(metaModel, metaGrads, outerLR: 0.001);
            }
        }
    }
}
```

#### Example 2: Newton's Method Optimization
```csharp
// examples/NewtonOptimization/NewtonExample.cs
public class NewtonExample
{
    public static void Run()
    {
        // Compare SGD vs Newton's method on Rosenbrock function

        // Define Rosenbrock function: f(x, y) = (a-x)² + b(y-x²)²
        Tensor Rosenbrock(Tensor[] params)
        {
            double a = 1.0, b = 100.0;
            var x = params[0];
            var y = params[1];
            return Math.Pow(a - x, 2) + b * Math.Pow(y - Math.Pow(x, 2), 2);
        }

        // Newton's method using Hessian
        void NewtonOptimization(Tensor[] parameters, int iterations)
        {
            var optimizer = new NewtonOptimizer(learningRate: 1.0, damping: 1e-4);

            for (int i = 0; i < iterations; i++)
            {
                var loss = Rosenbrock(parameters);
                var grads = Autograd.Gradient(loss, parameters);

                // Newton's method uses Hessian to compute update direction
                var update = optimizer.ComputeUpdateStep(loss, grads, parameters);
                UpdateParameters(parameters, update);

                Console.WriteLine($"Iteration {i}: Loss = {loss}");
            }
        }

        // SGD for comparison
        void SGDOptimization(Tensor[] parameters, int iterations)
        {
            var optimizer = new SGDOptimizer(learningRate: 0.001);

            for (int i = 0; i < iterations; i++)
            {
                var loss = Rosenbrock(parameters);
                var grads = Autograd.Gradient(loss, parameters);
                UpdateParameters(parameters, grads * optimizer.LearningRate);

                Console.WriteLine($"Iteration {i}: Loss = {loss}");
            }
        }

        // Compare convergence
        Console.WriteLine("Newton's Method:");
        var paramsNewton = InitializeParameters();
        NewtonOptimization(paramsNewton, 20);

        Console.WriteLine("\nSGD:");
        var paramsSGD = InitializeParameters();
        SGDOptimization(paramsSGD, 10000);
    }
}
```

#### Example 3: Neural ODEs
```csharp
// examples/NeuralODE/NeuralODEExample.cs
public class NeuralODEExample
{
    public static void Run()
    {
        // Neural ODE: Parameterize dynamics with neural network
        // Solve using higher-order derivatives for accurate integration

        Model dynamics = new MLP(...);

        // ODE solver using automatic differentiation
        Tensor[] SolveODE(Tensor[] initialConditions, double t0, double t1, int steps)
        {
            var dt = (t1 - t0) / steps;
            var state = initialConditions;

            for (int i = 0; i < steps; i++)
            {
                var t = t0 + i * dt;

                // 4th-order Runge-Kutta requires higher-order derivatives
                var k1 = dynamics(state);
                var k2 = dynamics(state + 0.5 * dt * k1);
                var k3 = dynamics(state + 0.5 * dt * k2);
                var k4 = dynamics(state + dt * k3);

                state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);
            }

            return state;
        }

        // Train Neural ODE
        void TrainNeuralODE(Model dynamics, Tensor[] trainingData)
        {
            var optimizer = new Adam(learningRate: 0.001);

            foreach (var (initialState, targetState) in trainingData)
            {
                var predictedState = SolveODE(initialState, t0: 0, t1: 1, steps: 100);
                var loss = MSE(predictedState, targetState);

                var grads = Autograd.Gradient(loss, dynamics.Parameters);
                optimizer.Update(dynamics.Parameters, grads);
            }
        }
    }
}
```

#### Example 4: Sharpness Minimization
```csharp
// examples/SharpnessMinimization/SharpnessExample.cs
public class SharpnessExample
{
    public static void Run()
    {
        // Minimize sharpness of minima to improve generalization
        // Use Hessian eigenvalues to measure sharpness

        Model model = new MLP(...);
        Tensor[] trainingData = LoadTrainingData();

        // Standard training
        TrainModel(model, trainingData);

        // Sharpness-aware training
        SharpnessAwareTrain(model, trainingData);

        void SharpnessAwareTrain(Model model, Tensor[] trainingData)
        {
            var optimizer = new Adam(learningRate: 0.001);

            foreach (var batch in trainingData)
            {
                // Compute loss at current parameters
                var loss = ComputeLoss(model, batch);

                // Perturb parameters in direction of largest Hessian eigenvalue
                var topEigenvalue = ComputeTopEigenvalue(loss, model.Parameters);
                var topEigenvector = ComputeTopEigenvector(loss, model.Parameters);
                var perturbedParams = model.Parameters + 0.01 * topEigenvector;

                // Compute loss at perturbed parameters
                var perturbedLoss = ComputeLoss(model, batch);

                // Update to minimize both losses (sharpness-aware)
                var combinedLoss = loss + 0.1 * perturbedLoss;
                var grads = Autograd.Gradient(combinedLoss, model.Parameters);

                optimizer.Update(model.Parameters, grads);
            }
        }

        double ComputeTopEigenvalue(Tensor loss, Tensor[] parameters)
        {
            // Use power iteration with HVP
            return PowerIteration(v => Autograd.HessianVectorProduct(loss, parameters, v));
        }
    }
}
```

#### Example 5: Adversarial Robustness
```csharp
// examples/AdversarialRobustness/AdversarialExample.cs
public class AdversarialExample
{
    public static void Run()
    {
        // Generate adversarial examples using curvature information
        // Use Hessian to find most sensitive input directions

        Model model = new CNN(...);
        Tensor input = LoadImage();
        Tensor trueLabel = LoadLabel();

        // Clean prediction
        var cleanPrediction = model.Forward(input);
        var cleanLoss = CrossEntropy(cleanPrediction, trueLabel);

        // Adversarial attack using Hessian
        Tensor GenerateAdversarialExample(Tensor input, Tensor trueLabel, double epsilon)
        {
            var loss = CrossEntropy(model.Forward(input), trueLabel);

            // Use Hessian to find most sensitive direction
            var hessian = Autograd.Hessian(loss, input);

            // Compute top eigenvector of Hessian
            var topEigenvector = ComputeTopEigenvector(hessian);

            // Perturb input in most sensitive direction
            var adversarialInput = input + epsilon * topEigenvector;

            return adversarialInput;
        }

        // Compare robustness
        var adversarialInput = GenerateAdversarialExample(input, trueLabel, epsilon: 0.01);
        var adversarialPrediction = model.Forward(adversarialInput);

        Console.WriteLine($"Clean prediction: {ArgMax(cleanPrediction)}");
        Console.WriteLine($"Adversarial prediction: {ArgMax(adversarialPrediction)}");
    }
}
```

## Implementation Tasks

### Phase 1: Core Examples
1. Implement MAML example with gradient-of-gradient
2. Implement Newton's method optimization example
3. Implement Neural ODE example with higher-order integration
4. Add basic documentation and comments

### Phase 2: Advanced Examples
1. Implement sharpness minimization example
2. Implement adversarial robustness example
3. Implement natural gradient example
4. Add comprehensive explanations

### Phase 3: Tutorials
1. Write tutorial on Jacobian computation and use cases
2. Write tutorial on Hessian computation and HVP
3. Write tutorial on higher-order derivatives for meta-learning
4. Add best practices and common pitfalls

### Phase 4: Documentation
1. Add README for each example
2. Add inline documentation with mathematical explanations
3. Add performance tips and optimization guidelines
4. Add troubleshooting guide

## Testing Requirements

### Example Validation
- All examples compile and run without errors
- Examples produce correct results (verify against known solutions)
- Examples demonstrate clear performance improvements where applicable

### Tutorial Completeness
- All steps in tutorials are executable
- Tutorials include expected outputs
- Tutorials explain mathematical concepts clearly

## Dependencies
- All previous implementation specs
- Example datasets (can be synthetic)
- Visualization utilities (optional)

## Success Criteria
- All 6 major examples implemented and working
- Each example includes clear documentation
- Tutorials cover key use cases comprehensively
- Examples demonstrate performance improvements
- Code is well-commented and educational

## Notes for Coder
- Focus on educational value - code should be clear and well-commented
- Use synthetic data where possible to avoid external dependencies
- Add performance comparisons where relevant
- Include expected outputs in comments
- Explain the mathematical concepts in the code
- Add error handling for common issues
- Keep examples self-contained and runnable
- Add tips for optimization and common pitfalls
- Consider adding Jupyter notebook versions for interactive exploration
