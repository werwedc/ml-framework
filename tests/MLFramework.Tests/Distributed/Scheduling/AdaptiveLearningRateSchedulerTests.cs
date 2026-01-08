using Microsoft.VisualStudio.TestTools.UnitTesting;
using MachineLearning.Distributed.Enums;
using MachineLearning.Distributed.Scheduling;
using System.Linq;

namespace MLFramework.Tests.Distributed.Scheduling;

[TestClass]
public class AdaptiveLearningRateSchedulerTests
{
    [TestMethod]
    public void Constructor_ValidParameters_CreatesScheduler()
    {
        // Act
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);

        // Assert
        Assert.IsNotNull(scheduler);
        Assert.AreEqual(0.01f, scheduler.CurrentLearningRate);
        Assert.AreEqual(4, scheduler.CurrentWorkerCount);
    }

    [TestMethod]
    public void Constructor_InvalidWorkerCount_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 0, 0.01f));

        Assert.ThrowsException<ArgumentException>(() =>
            new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, -1, 0.01f));
    }

    [TestMethod]
    public void Constructor_InvalidLearningRate_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0));

        Assert.ThrowsException<ArgumentException>(() =>
            new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, -0.01f));
    }

    [TestMethod]
    public void AdaptLearningRate_LinearStrategy_ScalesProportionally()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);

        // Act - Scale up to 8 workers
        var newLR = scheduler.AdaptLearningRate(4, 8, 0.01f);

        // Assert
        Assert.AreEqual(0.02f, newLR, 0.0001f);
        Assert.AreEqual(0.02f, scheduler.CurrentLearningRate);
        Assert.AreEqual(8, scheduler.CurrentWorkerCount);
    }

    [TestMethod]
    public void AdaptLearningRate_LinearStrategy_ScaleDown()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 8, 0.02f);

        // Act - Scale down to 2 workers
        var newLR = scheduler.AdaptLearningRate(8, 2, 0.02f);

        // Assert
        Assert.AreEqual(0.005f, newLR, 0.0001f);
        Assert.AreEqual(0.005f, scheduler.CurrentLearningRate);
        Assert.AreEqual(2, scheduler.CurrentWorkerCount);
    }

    [TestMethod]
    public void AdaptLearningRate_SquareRootStrategy_ScalesWithSqrt()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.SquareRoot, 4, 0.01f);

        // Act - Scale up to 16 workers (4x)
        var newLR = scheduler.AdaptLearningRate(4, 16, 0.01f);

        // Assert - Should be 2x (sqrt of 4x)
        Assert.AreEqual(0.02f, newLR, 0.0001f);
    }

    [TestMethod]
    public void AdaptLearningRate_SquareRootStrategy_MoreConservative()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.SquareRoot, 4, 0.01f);

        // Act
        var sqrtLR = scheduler.AdaptLearningRate(4, 16, 0.01f);
        var linearLR = 0.01f * 4; // Direct scaling

        // Assert - Square root should be more conservative
        Assert.IsTrue(sqrtLR < linearLR);
    }

    [TestMethod]
    public void AdaptLearningRate_NoneStrategy_KeepsConstant()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.None, 4, 0.01f);

        // Act - Scale to 8 workers
        var newLR = scheduler.AdaptLearningRate(4, 8, 0.01f);

        // Assert
        Assert.AreEqual(0.01f, newLR, 0.0001f);
    }

    [TestMethod]
    public void AdaptLearningRate_InvalidNewWorkerCount_ThrowsException()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);

        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            scheduler.AdaptLearningRate(4, 0, 0.01f));
    }

    [TestMethod]
    public void AdaptLearningRate_InvalidOldWorkerCount_ThrowsException()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);

        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            scheduler.AdaptLearningRate(0, 8, 0.01f));
    }

    [TestMethod]
    public void GetTargetLearningRate_Linear_ReturnsCorrectTarget()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);

        // Act
        var targetLR = scheduler.GetTargetLearningRate(8);

        // Assert
        Assert.AreEqual(0.02f, targetLR, 0.0001f);
    }

    [TestMethod]
    public void GetTargetLearningRate_DoesNotUpdateState()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);

        // Act
        var targetLR = scheduler.GetTargetLearningRate(8);

        // Assert
        Assert.AreEqual(0.01f, scheduler.CurrentLearningRate);
        Assert.AreEqual(4, scheduler.CurrentWorkerCount);
    }

    [TestMethod]
    public void TransitionLearningRate_ValidSteps_ReturnsCorrectSequence()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);
        var oldLR = 0.01f;
        var newLR = 0.02f;
        var steps = 5;

        // Act
        var transition = scheduler.TransitionLearningRate(oldLR, newLR, steps).ToList();

        // Assert
        Assert.AreEqual(steps, transition.Count);
        Assert.AreEqual(oldLR, transition[0], 0.0001f);
        Assert.AreEqual(newLR, transition[^1], 0.0001f);
    }

    [TestMethod]
    public void TransitionLearningRate_SingleStep_ReturnsNewLR()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);

        // Act
        var transition = scheduler.TransitionLearningRate(0.01f, 0.02f, 1).ToList();

        // Assert
        Assert.AreEqual(1, transition.Count);
        Assert.AreEqual(0.02f, transition[0], 0.0001f);
    }

    [TestMethod]
    public void TransitionLearningRate_InvalidSteps_ThrowsException()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);

        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            scheduler.TransitionLearningRate(0.01f, 0.02f, 0).ToList());

        Assert.ThrowsException<ArgumentException>(() =>
            scheduler.TransitionLearningRate(0.01f, 0.02f, -1).ToList());
    }

    [TestMethod]
    public void TransitionLearningRate_InterpolatesCorrectly()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);
        var steps = 11; // Includes 0.0 and 1.0

        // Act
        var transition = scheduler.TransitionLearningRate(0.01f, 0.03f, steps).ToList();

        // Assert - Middle value should be 0.02f
        Assert.AreEqual(0.02f, transition[5], 0.0001f);
    }

    [TestMethod]
    public void Reset_RestoresInitialState()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);
        scheduler.AdaptLearningRate(4, 8, 0.01f);

        // Act
        scheduler.Reset();

        // Assert
        Assert.AreEqual(0.01f, scheduler.CurrentLearningRate);
        Assert.AreEqual(4, scheduler.CurrentWorkerCount);
    }

    [TestMethod]
    public void MultipleAdaptations_MaintainCorrectRatio()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);

        // Act - Scale up, then down
        var lr1 = scheduler.AdaptLearningRate(4, 8, 0.01f);
        var lr2 = scheduler.AdaptLearningRate(8, 4, lr1);

        // Assert
        Assert.AreEqual(0.01f, lr2, 0.0001f);
    }
}

[TestClass]
public class LearningRateTransitionManagerTests
{
    [TestMethod]
    public void Constructor_ValidParameters_CreatesManager()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);

        // Act
        var manager = new LearningRateTransitionManager(scheduler, 100);

        // Assert
        Assert.IsNotNull(manager);
        Assert.IsFalse(manager.IsTransitioning);
        Assert.AreEqual(0.0f, manager.TransitionProgress);
    }

    [TestMethod]
    public void Constructor_NullScheduler_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() =>
            new LearningRateTransitionManager(null!, 100));
    }

    [TestMethod]
    public void Constructor_InvalidTransitionSteps_ThrowsException()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);

        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            new LearningRateTransitionManager(scheduler, 0));

        Assert.ThrowsException<ArgumentException>(() =>
            new LearningRateTransitionManager(scheduler, -1));
    }

    [TestMethod]
    public void StartTransition_StartsTransition()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);
        var manager = new LearningRateTransitionManager(scheduler, 10);

        // Act
        manager.StartTransition(4, 8, 0.01f);

        // Assert
        Assert.IsTrue(manager.IsTransitioning);
        Assert.AreEqual(0.0f, manager.TransitionProgress);
    }

    [TestMethod]
    public void GetCurrentLearningRate_ReturnsInterpolatedValue()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);
        var manager = new LearningRateTransitionManager(scheduler, 3);
        manager.StartTransition(4, 8, 0.01f);

        // Act
        var lr1 = manager.GetCurrentLearningRate();
        var lr2 = manager.GetCurrentLearningRate();

        // Assert
        Assert.AreEqual(0.01f, lr1, 0.0001f); // First step
        Assert.AreEqual(0.015f, lr2, 0.0001f); // Second step
    }

    [TestMethod]
    public void GetCurrentLearningRate_CompletesTransition()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);
        var manager = new LearningRateTransitionManager(scheduler, 3);
        manager.StartTransition(4, 8, 0.01f);

        // Act
        manager.GetCurrentLearningRate();
        manager.GetCurrentLearningRate();
        var lr3 = manager.GetCurrentLearningRate();

        // Assert
        Assert.AreEqual(0.02f, lr3, 0.0001f); // Final step
        Assert.IsFalse(manager.IsTransitioning);
    }

    [TestMethod]
    public void GetCurrentLearningRate_BeforeTransition_ReturnsSchedulerLR()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);
        var manager = new LearningRateTransitionManager(scheduler, 10);

        // Act
        var lr = manager.GetCurrentLearningRate();

        // Assert
        Assert.AreEqual(0.01f, lr, 0.0001f);
    }

    [TestMethod]
    public void CompleteTransition_SkipsToTargetLR()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);
        var manager = new LearningRateTransitionManager(scheduler, 100);
        manager.StartTransition(4, 8, 0.01f);
        manager.GetCurrentLearningRate(); // Advance one step

        // Act
        manager.CompleteTransition();

        // Assert
        Assert.IsFalse(manager.IsTransitioning);
    }

    [TestMethod]
    public void Reset_ClearsTransitionState()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);
        var manager = new LearningRateTransitionManager(scheduler, 10);
        manager.StartTransition(4, 8, 0.01f);
        manager.GetCurrentLearningRate();

        // Act
        manager.Reset();

        // Assert
        Assert.IsFalse(manager.IsTransitioning);
        Assert.AreEqual(0.0f, manager.TransitionProgress);
    }

    [TestMethod]
    public void TransitionProgress_TracksCorrectly()
    {
        // Arrange
        var scheduler = new AdaptiveLearningRateScheduler(AdaptationStrategy.Linear, 4, 0.01f);
        var manager = new LearningRateTransitionManager(scheduler, 10);
        manager.StartTransition(4, 8, 0.01f);

        // Act
        manager.GetCurrentLearningRate(); // 1/10

        // Assert
        Assert.AreEqual(0.1f, manager.TransitionProgress, 0.01f);
    }
}
