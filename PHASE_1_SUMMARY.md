# Phase 1: Concurrency Scaling Validation — Benchmark Results

**Model:** Qwen/Qwen3-30B-A3B  
**Input Tokens:** 8,192  
**Output Tokens:** 64  
**Batches:** 3  
**Total Traces:** 1536

## Timing Breakdown (Mean, Mutually Exclusive Buckets)

| Concurrency | CPU Measured | Client/Network Wait | Server/Header Wait | GPU Time | Total | CPU% | Wait% | GPU% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **512** | 0.1ms | 170.2ms | 3570.1ms | 143736.1ms | 147476.6ms | 0.0% | 2.5% | 97.5% |

## Latency Breakdown (P95, ms)

| Concurrency | Serialize ($P_{95}$) | Client/Network Wait ($P_{95}$) | Server/Header Wait ($P_{95}$) | Prefill ($P_{95}$) | Decode ($P_{95}$) | E2E ($P_{95}$) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **512** | 0.2 | 241.0 | 6494.0 | 199544.1 | 45154.8 | 218095.9 | Breakdown |

## Detailed Statistics (ms)

| Concurrency | Metric | Mean | P50 | P95 | P99 | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **512** | Client Serialization | 0.15 | 0.15 | 0.16 | 0.19 | 0.12 | 1.09 |
| **512** | Semaphore Wait (Client Gate) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 |
| **512** | HTTP Connect (Queue+Net) | 3740.25 | 3665.30 | 6653.49 | 7002.16 | 294.38 | 7228.40 |
| **512** | HTTP Connector Pool Wait | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **512** | HTTP DNS Lookup | 0.15 | 0.00 | 0.00 | 0.00 | 0.00 | 80.18 |
| **512** | HTTP TCP Connect | 136.28 | 156.56 | 230.21 | 245.61 | 0.00 | 264.83 |
| **512** | HTTP Request Send | 33.80 | 10.81 | 130.45 | 142.02 | 0.74 | 145.52 |
| **512** | HTTP Response Headers Wait | 3570.08 | 3476.19 | 6493.98 | 6830.87 | 149.52 | 6991.91 |
| **512** | First SSE Byte to First Token | 0.02 | 0.02 | 0.02 | 0.03 | 0.01 | 0.04 |
| **512** | TTFT (Server Prefill) | 107462.16 | 107570.57 | 206309.91 | 214692.37 | 670.83 | 216867.77 |
| **512** | GPU Prefill (TTFT - 1stByte) | 103721.90 | 103893.36 | 199544.12 | 207911.38 | 336.53 | 210045.55 |
| **512** | GPU Decode | 40014.21 | 44401.53 | 45154.76 | 45952.77 | 1783.60 | 47303.12 |
| **512** | End-to-End | 147476.54 | 152408.53 | 218095.87 | 218535.81 | 43679.14 | 218651.52 |

## HTTP Subphase Breakdown (P95, ms)

| Concurrency | Total HTTP | Conn Pool | DNS | TCP Connect | Request Send | Response Headers |
| --- | --- | --- | --- | --- | --- | --- |
| **512** | 6653.5 | 0.0 | 0.0 | 230.2 | 130.5 | 6494.0 |

## Server-side OTel Breakdown (P95, ms)

| Concurrency | Client RespHdr Wait | Server Queue | Server TTFT | Model Prefill | Model Decode | Server E2E | Span Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **512** | 6494.0 | 192327.3 | 201837.1 | 2843.9 | 45157.7 | 217464.9 | 1425 |

## Server-side Span Timeline (P95 offset from scenario start, ms)

| Concurrency | Request Start | First Token | Inference End | Request End | Span Duration |
| --- | --- | --- | --- | --- | --- |
| **512** | 436797.1 | 575414.9 | 483900.2 | 621756.7 | 217464.9 |
