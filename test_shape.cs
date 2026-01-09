using RitterFramework.Core.Tensor;

var gradient = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 2, 2 });
var input = new Tensor(new float[] { 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f }, new[] { 2, 3 });

Console.WriteLine($"Gradient shape: [{string.Join(", ", gradient.Shape)}] sum={gradient.Shape.Sum()}");
Console.WriteLine($"Input shape: [{string.Join(", ", input.Shape)}] sum={input.Shape.Sum()}");
