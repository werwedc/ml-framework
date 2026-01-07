```

BenchmarkDotNet v0.14.0, Windows 10 (10.0.19045.6456/22H2/2022Update)
AMD Ryzen 7 3800X, 1 CPU, 16 logical and 8 physical cores
.NET SDK 9.0.302
  [Host]     : .NET 9.0.7 (9.0.725.31616), X64 RyuJIT AVX2
  Job-EPCXOS : .NET 9.0.7 (9.0.725.31616), X64 RyuJIT AVX2
  ShortRun   : .NET 9.0.7 (9.0.725.31616), X64 RyuJIT AVX2

WarmupCount=3  

```
| Method                     | Job        | IterationCount | LaunchCount | SequenceLength | BatchSize | Mean     | Error    | StdDev  | Gen0   | Allocated |
|--------------------------- |----------- |--------------- |------------ |--------------- |---------- |---------:|---------:|--------:|-------:|----------:|
| PagedAttention_SingleToken | Job-EPCXOS | 10             | Default     | 16             | 1         | 420.0 μs |  4.65 μs | 3.08 μs | 2.9297 |  24.74 KB |
| PagedAttention_SingleToken | ShortRun   | 3              | 1           | 16             | 1         | 402.2 μs | 33.76 μs | 1.85 μs | 2.9297 |  24.74 KB |
