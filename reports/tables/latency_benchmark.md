# CPU Inference Latency Benchmark Report

- **CPU**: Intel64 Family 6 Model 142 Stepping 12, GenuineIntel (8 threads)
- **RAM**: 15.81 GB
- **Batch Size**: 1 (Real-Time Edge/API)
- **Threads**: 4 intra-op threads

| Metric | End-to-End Latency (ms) | Pure ONNX Forward Pass (ms) |
| --- | --- | --- |
| **Mean** | 68.12 ms | 58.24 ms |
| **Median (p50)** | 65.61 ms | 55.57 ms |
| **p90** | 86.78 ms | - |
| **p95** | 92.32 ms | 78.61 ms |
| **Throughput** | **14.7 FPS** | **17.2 FPS** |

