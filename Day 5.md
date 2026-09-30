# Day 5 — LLM Hardware, Model Size, Memory, Quantization & Inference

## Beginner → Engineering Student Guide

> **Goal:** Understand what happens when you take a trained LLM and actually try to run it on a computer.

By the end of Day 5, you should understand:

- 7B / 13B / 70B
- Model memory and parameter size
- FP32, FP16, BF16, INT8 and INT4
- RAM vs VRAM
- GPUs and tensor operations
- Model loading
- Inference
- Latency, TTFT and throughput
- Tokens/sec
- KV cache
- Quantization
- CPU offloading
- Batching
- Prefill and decode
- Local LLMs
- Model serving
- Basic AI infrastructure

---

# 1. Day 4 → Day 5

On Day 4 we learned:

```text
Training data
    ↓
Prediction
    ↓
Loss
    ↓
Backpropagation
    ↓
Gradients
    ↓
Optimizer
    ↓
Parameter updates
    ↓
Repeat
    ↓
Trained model
```

Today we ask:

> **How do we actually run that trained model?**

That process is called **inference**.

---

# 2. Training vs Inference

## Training

The model learns.

```text
Data
 ↓
Prediction
 ↓
Loss
 ↓
Gradient
 ↓
Parameter update
```

The weights change.

## Inference

The model is used.

```text
Prompt
 ↓
Model
 ↓
Prediction
 ↓
Response
```

The model's weights normally stay fixed.

### Simple analogy

A student:

```text
Training:
Study → Practice → Mistakes → Correction → Learn

Inference:
Question → Answer
```

---

# 3. What Is a Trained Model?

A useful mental model is:

```text
Architecture
+
Learned parameters
=
Trained model
```

For example:

```text
Transformer architecture
+
7 billion learned parameters
=
7B model
```

---

# 4. What Does 7B Mean?

`7B` means approximately:

```text
7 billion parameters
```

Examples:

```text
1B  → ~1 billion
3B  → ~3 billion
7B  → ~7 billion
8B  → ~8 billion
13B → ~13 billion
70B → ~70 billion
```

The `B` means billion.

---

# 5. Why Does Parameter Count Matter?

Parameters are numerical values that require memory.

Therefore:

```text
More parameters
      ↓
More numbers to store
      ↓
More memory
```

And generally:

```text
More parameters
      ↓
More computation
      ↓
Greater hardware requirements
```

But remember:

> More parameters does **not** automatically mean a better model.

Architecture, data quality, training, optimization and other factors matter too.

---

# 6. What Is a Bit?

A bit can contain:

```text
0
```

or:

```text
1
```

With 2 bits:

```text
00
01
10
11
```

So there are 4 possible combinations.

With 8 bits:

```text
2^8 = 256
```

possible combinations.

---

# 7. What Is a Byte?

```text
8 bits = 1 byte
```

For simple calculations:

```text
1 KB ≈ 1,000 bytes
1 MB ≈ 1,000 KB
1 GB ≈ 1,000 MB
1 TB ≈ 1,000 GB
```

You may also encounter:

```text
KiB
MiB
GiB
TiB
```

which use binary multiples.

---

# 8. Why Do Bits Matter for LLMs?

Model parameters are stored using numerical formats.

Common examples:

```text
FP32
FP16
BF16
INT8
INT4
```

They use different amounts of memory.

The basic idea is:

```text
More bits per parameter
→ More memory

Fewer bits per parameter
→ Less memory
```

---

# 9. FP32

FP32 means:

> 32-bit floating point.

Approximately:

```text
32 bits = 4 bytes
```

Therefore a 7B model's weights would require approximately:

```text
7 billion × 4 bytes
≈ 28 GB
```

This is a **weight-only estimate**.

---

# 10. FP16

FP16 means:

> 16-bit floating point.

Approximately:

```text
16 bits = 2 bytes
```

A 7B model:

```text
7B × 2 bytes
≈ 14 GB
```

Again, this is approximately the memory for the weights alone.

---

# 11. BF16

BF16 means:

> Brain Floating Point 16-bit.

It uses 16 bits, so for a simple memory calculation:

```text
7B × 2 bytes
≈ 14 GB
```

FP16 and BF16 both use 16 bits but represent numbers differently.

BF16 is particularly useful in large-scale deep learning because it offers a large numerical range while reducing memory compared with FP32.

---

# 12. INT8

INT8 means:

> 8-bit integer representation.

Approximately:

```text
8 bits = 1 byte
```

A 7B model:

```text
7B × 1 byte
≈ 7 GB
```

---

# 13. INT4

INT4 means:

> 4-bit integer representation.

Approximately:

```text
4 bits = 0.5 byte
```

A 7B model:

```text
7B × 0.5 byte
≈ 3.5 GB
```

This is one reason 4-bit quantization is popular for local LLMs.

---

# 14. Quick Memory Table

For rough **weight-only** calculations:

| Format | Bits / parameter | Approx bytes / parameter |
|---|---:|---:|
| FP32 | 32 | 4 |
| FP16 | 16 | 2 |
| BF16 | 16 | 2 |
| INT8 | 8 | 1 |
| INT4 | 4 | 0.5 |

Formula:

```text
Approximate weight memory
=
Number of parameters
×
Bytes per parameter
```

---

# 15. 7B Model Example

### FP32

```text
7B × 4
≈ 28 GB
```

### FP16/BF16

```text
7B × 2
≈ 14 GB
```

### INT8

```text
7B × 1
≈ 7 GB
```

### INT4

```text
7B × 0.5
≈ 3.5 GB
```

---

# 16. 13B Model Example

### FP32

```text
13B × 4
≈ 52 GB
```

### FP16/BF16

```text
13B × 2
≈ 26 GB
```

### INT8

```text
13B × 1
≈ 13 GB
```

### INT4

```text
13B × 0.5
≈ 6.5 GB
```

---

# 17. 70B Model Example

### FP32

```text
70B × 4
≈ 280 GB
```

### FP16/BF16

```text
70B × 2
≈ 140 GB
```

### INT8

```text
70B × 1
≈ 70 GB
```

### INT4

```text
70B × 0.5
≈ 35 GB
```

Now you can see why 70B models require serious hardware.

---

# 18. Important: Weight Memory Is Not Total Memory

This is one of the most important engineering lessons.

If:

```text
7B INT4 weights ≈ 3.5 GB
```

you should **not** conclude:

> "The model only needs 3.5 GB RAM."

Inference also needs memory for things such as:

```text
KV cache
Activations
Temporary tensors
Runtime buffers
Framework overhead
Other processes
```

Therefore:

```text
Total inference memory
>
Weight memory
```

in practical deployments.

---

# 19. RAM

RAM is the computer's general-purpose working memory.

It is used for:

```text
Operating system
Applications
Browser
Databases
Model data
Intermediate computations
```

You may see machines with:

```text
16 GB RAM
32 GB RAM
64 GB RAM
128 GB RAM
```

---

# 20. VRAM

VRAM means:

> Video RAM.

It is memory associated with a GPU.

Discrete GPUs may have:

```text
8 GB
12 GB
16 GB
24 GB
48 GB
80 GB
```

or more.

GPU inference often benefits from keeping model weights and computation on GPU memory.

---

# 21. RAM vs VRAM

Simplified:

```text
CPU
 ↓
RAM
```

and:

```text
GPU
 ↓
VRAM
```

For a discrete GPU:

```text
RAM
 ↕
VRAM
 ↕
GPU compute
```

Moving data between CPU memory and GPU memory can introduce overhead.

---

# 22. Why GPU Memory Matters

Suppose:

```text
GPU VRAM = 8 GB
```

and:

```text
7B FP16 model ≈ 14 GB weights
```

The complete FP16 weights cannot fit in 8 GB.

Possible solutions:

```text
Quantize the model
Use a smaller model
Use CPU offloading
Use multiple GPUs
Use hardware with more memory
```

---

# 23. What Is a GPU?

GPU means:

> Graphics Processing Unit.

GPUs are extremely good at large amounts of parallel numerical computation.

That makes them useful for:

```text
Matrix multiplication
Tensor operations
Deep learning
LLM inference
LLM training
```

---

# 24. CPU vs GPU

## CPU

Good at:

```text
General-purpose computing
Operating system
Application logic
Branch-heavy operations
Sequential/control-oriented tasks
```

## GPU

Good at:

```text
Highly parallel numerical operations
Matrix operations
Tensor operations
Deep learning
```

A simple analogy:

```text
CPU = small number of highly capable workers

GPU = very large number of workers doing similar operations in parallel
```

---

# 25. What Is a Tensor?

A tensor is essentially a multidimensional array of numbers.

### Scalar

```text
5
```

### Vector

```text
[1, 2, 3]
```

### Matrix

```text
[
 [1, 2],
 [3, 4]
]
```

### Higher-dimensional tensor

```text
[
  [
    [1, 2],
    [3, 4]
  ],
  [
    [5, 6],
    [7, 8]
  ]
]
```

Deep-learning frameworks operate heavily on tensors.

---

# 26. Why Are GPUs Good at AI?

A simplified neural-network layer can involve:

```text
input × weights
```

which becomes a large matrix multiplication.

A GPU can perform many multiplication/addition operations in parallel.

Therefore:

```text
Neural network
→ lots of tensor operations
→ GPU is highly useful
```

---

# 27. What Are Tensor Cores?

Many modern AI GPUs contain specialized hardware for accelerating matrix/tensor operations.

For example:

```text
Tensor Cores
```

The beginner-level takeaway:

```text
Tensor Cores
→ specialized hardware
→ accelerates common AI calculations
```

---

# 28. What Is Model Loading?

Before inference, the model must be loaded.

Conceptually:

```text
Model file
   ↓
Storage
   ↓
RAM / VRAM / unified memory
   ↓
Inference runtime
   ↓
Model ready
```

For a large model, loading can take noticeable time.

---

# 29. What Is Inference?

Inference means:

> Using a trained model to generate a prediction.

Example:

```text
Prompt:
Explain Kubernetes.

       ↓

Trained LLM

       ↓

Generated answer
```

No normal parameter update occurs.

---

# 30. How Does LLM Generation Work?

Suppose:

```text
The sky is
```

The model predicts a next token.

Maybe:

```text
blue
```

Then the context becomes:

```text
The sky is blue
```

The model predicts another token.

Maybe:

```text
today
```

Then:

```text
The sky is blue today
```

This repeats.

Conceptually:

```text
Predict token
 ↓
Add token to context
 ↓
Predict next token
 ↓
Add token
 ↓
Repeat
```

---

# 31. What Is Autoregressive Generation?

Autoregressive generation means the model uses previously generated tokens as part of the input for future predictions.

Example:

```text
Token 1
 ↓
Token 2
 ↓
Token 3
 ↓
Token 4
 ↓
...
```

This is how many popular LLMs generate text.

---

# 32. What Is Latency?

Latency means:

> How long something takes.

For LLMs, you might measure:

```text
Time to first token
```

and:

```text
Total response time
```

---

# 33. TTFT

TTFT means:

> **Time To First Token**

Example:

```text
Request sent
     ↓
0.8 seconds
     ↓
First token appears
```

Then:

```text
TTFT = ~0.8 seconds
```

For interactive AI applications, lower TTFT often makes the system feel more responsive.

---

# 34. Tokens per Second

Suppose:

```text
Generation speed = 20 tokens/sec
```

That means the model generates approximately:

```text
20 output tokens every second
```

If a response has roughly:

```text
200 tokens
```

then generation itself might take approximately:

```text
200 / 20
=
10 seconds
```

This is a simplified calculation.

---

# 35. Latency vs Throughput

These are different.

## Latency

How long one request takes.

```text
Request → Response
       2 seconds
```

## Throughput

How much work the system can handle over time.

For example:

```text
100 requests/minute
```

or:

```text
1,000 generated tokens/sec
```

depending on the metric.

---

# 36. Restaurant Analogy

Think about a restaurant.

### Latency

How long one customer waits for food.

### Throughput

How many meals the restaurant serves per hour.

A restaurant can have:

```text
Excellent individual service
```

but:

```text
Low number of meals served
```

or:

```text
High throughput
```

with longer individual waiting times.

The same tradeoff exists in AI serving.

---

# 37. What Is Batching?

Suppose several users send requests:

```text
User A
User B
User C
User D
```

The inference system may process them together:

```text
A ┐
B ├──→ Batch → GPU
C │
D ┘
```

This can improve GPU utilization and throughput.

---

# 38. Why Not Always Use Huge Batches?

Because larger batches can require more:

```text
Memory
Scheduling
Compute
```

and can affect latency.

Therefore production systems balance:

```text
Latency
+
Throughput
+
Memory
+
Cost
```

---

# 39. What Is KV Cache?

KV cache is one of the most important concepts in LLM inference.

In transformer attention, the model uses:

```text
K = Keys
V = Values
```

During autoregressive generation, useful key/value information from previous tokens can be cached.

This avoids unnecessarily repeating some calculations.

---

# 40. KV Cache Analogy

Imagine reading a long technical document.

Without notes:

```text
Every time you answer a question,
reread the entire document.
```

With notes:

```text
Keep useful information from earlier reading.
```

KV cache is somewhat analogous to computational notes.

---

# 41. Why Is KV Cache Important?

It improves generation efficiency.

But it also consumes memory.

Therefore total memory can involve:

```text
Model weights
+
KV cache
+
Activations
+
Runtime buffers
```

As context length and concurrency increase, KV cache can become a significant memory consumer.

---

# 42. What Is Context Length?

The context is the information the model can consider for a request.

It can contain:

```text
System instructions
+
Conversation history
+
User prompt
+
Retrieved documents
+
Tool results
```

The model's context limit is measured in tokens.

For example, a model may support a context size on the order of:

```text
8K
32K
128K
```

depending on the model.

---

# 43. Why Does Long Context Cost More?

Suppose:

```text
Prompt = 100 tokens
```

versus:

```text
Prompt = 100,000 tokens
```

The second requires substantially more input processing.

During generation, maintaining information about a long context can also increase KV-cache memory.

Therefore:

```text
Long context
→ more compute/memory requirements
```

---

# 44. What Is Prefill?

When the model receives an existing prompt, it first processes that input context.

This phase is commonly called:

> **Prefill**

Conceptually:

```text
Long prompt
 ↓
Process input tokens
 ↓
Prepare internal state
 ↓
Start generation
```

---

# 45. What Is Decode?

After prefill, the model generates output tokens.

This phase is commonly called:

> **Decode**

Conceptually:

```text
Generate token 1
 ↓
Generate token 2
 ↓
Generate token 3
 ↓
...
```

So:

```text
Inference
=
Prefill + Decode
```

---

# 46. Why Are Prefill and Decode Different?

Prefill processes many input tokens.

Decode generally generates one new token at a time.

Therefore they have different performance characteristics.

This matters when optimizing LLM inference.

---

# 47. Example

Suppose:

```text
Prompt = 10,000 tokens
Response = 200 tokens
```

The system must process:

```text
10,000 input tokens
```

during prefill.

Then generate:

```text
200 output tokens
```

during decode.

So a long prompt can affect latency even when the final answer is short.

---

# 48. What Is Quantization?

Quantization means representing model values with fewer bits.

For example:

```text
FP16
 ↓
INT8
```

or:

```text
FP16
 ↓
INT4
```

The objective is generally to reduce:

```text
Memory
Storage
Bandwidth
Sometimes inference cost
```

while retaining as much model quality as possible.

---

# 49. Simple Quantization Analogy

Suppose you record someone's height.

High precision:

```text
175.238491 cm
```

Lower precision:

```text
175 cm
```

The second is less precise but may be perfectly sufficient for some applications.

Quantization applies a related concept to model numbers.

---

# 50. Does Quantization Always Reduce Quality?

It can.

The tradeoff is approximately:

```text
Lower precision
      ↓
Lower memory
      ↓
Potentially easier/faster inference
      ↓
Potential quality/numerical impact
```

The actual impact depends on:

```text
Model
Quantization method
Precision
Calibration
Runtime
Task
```

Good quantization can preserve a large amount of model quality.

---

# 51. Why Is 4-bit Popular?

Because the memory reduction is dramatic.

For a 7B model:

```text
FP16:
~14 GB

INT4:
~3.5 GB
```

That's approximately a:

```text
4× reduction
```

in the simple weight-memory calculation.

This can make local inference possible on hardware that couldn't hold the FP16 weights.

---

# 52. Very Important: 7B vs 4-bit

These mean different things.

```text
7B
→ Number of parameters

4-bit
→ Numerical representation of parameters
```

Therefore:

```text
7B 4-bit
```

means:

```text
~7 billion parameters
represented at approximately 4 bits each
```

It does **not** mean:

```text
4 billion parameters
```

---

# 53. Example: 7B FP16 vs 13B INT4

### 7B FP16

```text
7 × 2
≈ 14 GB
```

### 13B INT4

```text
13 × 0.5
≈ 6.5 GB
```

So:

```text
13B INT4
```

can have more parameters but require less weight memory than:

```text
7B FP16
```

This is why parameter count alone isn't enough when discussing hardware requirements.

---

# 54. What Is CPU Offloading?

Suppose:

```text
GPU VRAM = 8 GB
```

but the model needs more.

Some inference runtimes can keep some data in system RAM while using the GPU for other parts.

Conceptually:

```text
System RAM
 ├── Model part A
 └── Model part B

GPU VRAM
 └── Model part C
```

This is called offloading.

---

# 55. Why Can Offloading Be Slower?

Data may need to move between:

```text
RAM
 ↕
GPU memory
```

Transfers can become a bottleneck.

Therefore:

```text
More GPU-resident computation
→ often faster

More offloading
→ may be slower
```

But offloading can make a model runnable when it otherwise would not fit.

---

# 56. Multiple GPUs

If a model is too large for one GPU, multiple GPUs can be used.

Conceptually:

```text
GPU 1
GPU 2
GPU 3
GPU 4
```

The model or computation can be distributed across them.

---

# 57. Data Parallelism

Different GPUs process different training batches.

```text
              Training data
                   ↓
       ┌───────────┼───────────┐
       ↓           ↓           ↓
     GPU 1       GPU 2       GPU 3
       ↓           ↓           ↓
    Batch A      Batch B      Batch C
```

This is particularly important during training.

---

# 58. Model Parallelism

The model itself can be distributed.

Simplified:

```text
GPU 1 → Layers 1–10
GPU 2 → Layers 11–20
GPU 3 → Layers 21–30
```

This helps when the model cannot fit on one GPU.

---

# 59. Tensor Parallelism

Large tensor operations can also be split across GPUs.

Simplified:

```text
Large matrix
     ↓
   Split
  ↙     ↘
GPU 1   GPU 2
  ↓       ↓
Partial  Partial
result   result
  ↘       ↙
   Combined
```

The real implementation is much more sophisticated.

---

# 60. Why Is Networking Important?

When multiple GPUs work together, they must communicate.

For large AI systems, communication can become a bottleneck.

Therefore infrastructure engineers care about:

```text
GPU compute
+
GPU memory
+
Memory bandwidth
+
Network bandwidth
+
Network latency
+
Storage
```

---

# 61. What Is Memory Bandwidth?

Memory bandwidth describes how quickly data can move between memory and compute hardware.

For example:

```text
High bandwidth
→ more data can be moved per second
```

This matters because LLM inference repeatedly reads large amounts of model data.

---

# 62. Why Can Inference Be Memory-Bandwidth Bound?

A simplified process is:

```text
Read model weights
      ↓
Compute
      ↓
Read more weights
      ↓
Compute
      ↓
...
```

If the processor is waiting for data from memory, adding more compute capacity may not solve the bottleneck.

Therefore:

```text
Compute
+
Memory bandwidth
```

both matter.

---

# 63. What Is a Model Runtime?

A runtime is software that executes the model.

Examples you will encounter include:

```text
PyTorch
Transformers
llama.cpp
vLLM
TensorRT-LLM
```

Different runtimes have different goals and hardware optimizations.

The important mental model is:

```text
Model weights
+
Architecture
+
Runtime
+
Hardware
=
Inference
```

---

# 64. What Is llama.cpp?

`llama.cpp` is a widely used project for efficient local inference of compatible language models.

It is important in the local-LLM ecosystem because it supports:

```text
CPU inference
GPU acceleration
Quantized models
Local execution
```

The engineering lesson is:

> A model file alone is not enough. You also need software capable of executing it.

---

# 65. What Is vLLM?

vLLM is an inference and serving engine designed for efficient LLM serving.

It focuses on things such as:

```text
High throughput
Efficient memory management
Batching
Production serving
```

It is particularly useful when serving models to many users.

---

# 66. What Is GGUF?

GGUF is a model file format commonly used in the llama.cpp/local-LLM ecosystem.

It can contain:

```text
Model metadata
Weights
Quantization information
Other runtime-related information
```

You may see files such as:

```text
model.Q4_K_M.gguf
```

For a beginner:

> GGUF is a model format commonly used for efficient local LLM inference.

---

# 67. What Does Q4 Mean?

In many local model names:

```text
Q4
```

indicates approximately:

```text
4-bit quantization
```

Additional letters can describe the particular quantization scheme.

For now:

```text
Q4
→ roughly 4-bit quantized model
```

---

# 68. Local LLMs

A local LLM runs on your own hardware.

Conceptually:

```text
Model file
 ↓
Local runtime
 ↓
CPU/GPU/accelerator
 ↓
Prompt
 ↓
Inference
 ↓
Response
```

Benefits can include:

```text
Privacy
Offline operation
Experimentation
Lower API dependence
Learning
```

---

# 69. API-Based LLMs

With an API:

```text
Your application
      ↓
Internet
      ↓
Provider
      ↓
Large model
      ↓
Response
```

Advantages can include:

```text
No local GPU requirement
Easy scaling
Access to very large models
Managed infrastructure
```

Tradeoffs can include:

```text
API cost
Network latency
Rate limits
Vendor dependency
Data-governance considerations
```

---

# 70. Local vs API

| Factor | Local | API |
|---|---|---|
| Hardware | You provide it | Provider provides it |
| Setup | More engineering | Usually easier |
| Scaling | Your responsibility | Provider handles much of it |
| Privacy control | Potentially high | Depends on provider/setup |
| Cost | Hardware upfront | Usually usage-based |
| Model choice | Local-compatible models | Provider catalog |
| Learning | Excellent | Excellent |

---

# 71. Why Can a Mac Run Local LLMs?

Modern Apple-silicon Macs use a unified-memory architecture.

A simplified picture:

```text
           Unified Memory
          /      |                ↓       ↓        ↓
       CPU      GPU    Accelerators
```

This differs from the traditional:

```text
CPU → RAM
GPU → separate VRAM
```

architecture.

A machine with enough unified memory can therefore run models that would not fit inside a small dedicated GPU's VRAM.

But:

```text
Can run
≠
Runs as fast as a high-end data-center GPU
```

---

# 72. What Is Unified Memory?

In a unified-memory architecture, CPU and GPU/accelerators can access a shared memory pool.

Conceptually:

```text
          Shared Memory
        /       |              CPU     GPU    Accelerator
```

This can be useful for local AI workloads.

---

# 73. What Should You Check Before Running a Local Model?

Use this checklist:

```text
1. Parameter count
2. Quantization
3. Weight memory
4. Available RAM/VRAM/unified memory
5. Context length
6. KV cache requirements
7. Runtime compatibility
8. CPU/GPU/accelerator capability
9. Expected tokens/sec
10. Other applications using memory
```

---

# 74. Example Hardware Decision

Suppose:

```text
Available GPU VRAM = 8 GB
```

and you want:

```text
7B FP16
```

Weight estimate:

```text
~14 GB
```

Conclusion:

```text
It cannot fit entirely in 8 GB VRAM.
```

Options:

```text
7B INT4
Smaller model
CPU offloading
Multiple GPUs
More-memory GPU
```

---

# 75. Example: 16 GB Unified Memory

Suppose a machine has:

```text
16 GB unified memory
```

and you want:

```text
7B INT4
```

Weight estimate:

```text
~3.5 GB
```

This may be feasible, depending on:

```text
Runtime
Context
KV cache
Operating system usage
Other applications
Model format
```

Do not assume the entire 16 GB is available to the model.

---

# 76. Why Model Fit Is Not Enough

Suppose:

```text
Available memory = 8 GB
Model weights = 7.8 GB
```

It may still fail or perform poorly because the system also needs:

```text
KV cache
Runtime memory
OS
Application memory
Temporary buffers
```

Therefore:

> **"Fits on paper" is not the same as "runs comfortably."**

---

# 77. What Is Prefill vs Decode Performance?

A useful production distinction:

```text
Prefill
→ processing the prompt

Decode
→ generating output tokens
```

A system can be fast at one and slower at the other.

This is important for:

```text
Chatbots
RAG
Long documents
Coding assistants
Agents
```

---

# 78. Why RAG Affects Inference Performance

Suppose a user asks:

```text
What is our leave policy?
```

RAG retrieves:

```text
2,500 tokens
```

Your original question might be:

```text
20 tokens
```

The LLM may therefore process approximately:

```text
2,520 tokens
```

plus:

```text
System prompt
Conversation history
Other context
```

So RAG affects:

```text
Input processing
Latency
Token usage
Memory
Cost
```

---

# 79. Good RAG Is Not "Retrieve Everything"

A poor RAG system might do:

```text
Question
 ↓
Retrieve 100 documents
 ↓
Put everything into prompt
 ↓
LLM
```

A better system aims for:

```text
Question
 ↓
Retrieve candidates
 ↓
Rank/re-rank
 ↓
Select relevant chunks
 ↓
LLM
```

This reduces unnecessary context.

---

# 80. What Is Reranking?

Suppose retrieval produces:

```text
20 candidate chunks
```

A reranker can score them for relevance:

```text
20 candidates
      ↓
Reranker
      ↓
Top 5
      ↓
LLM
```

This can improve the quality of the context supplied to the model.

---

# 81. What Is Model Serving?

Serving means making the model available to applications/users.

A production system might look like:

```text
Users
  ↓
API Gateway
  ↓
Load Balancer
  ↓
Inference Server
  ↓
GPU Workers
  ↓
LLM
```

Around this you may also have:

```text
Authentication
Rate limiting
Monitoring
Logging
Autoscaling
Caching
Evaluation
```

---

# 82. What Is Autoscaling?

Traffic changes.

For example:

```text
Morning:
10 requests/sec

Peak:
100 requests/sec
```

A production system may increase the number of inference workers.

Conceptually:

```text
Low traffic → 2 workers
High traffic → 10 workers
```

This is autoscaling.

---

# 83. What Is Rate Limiting?

Suppose one client sends:

```text
10,000 requests/sec
```

That could overwhelm the service.

Rate limiting may enforce something like:

```text
100 requests/minute/user
```

The exact limit depends on the application.

This is normal backend engineering applied to AI systems.

---

# 84. What Is Caching?

If the same request is repeatedly made, caching can sometimes reduce:

```text
Latency
Compute
Cost
```

But generated responses may depend on:

```text
User
Context
Time
Retrieved information
Model version
```

so caching must be designed carefully.

---

# 85. What Is GPU Utilization?

GPU utilization gives an indication of how much the GPU is being used.

Low utilization can indicate:

```text
Poor batching
CPU bottleneck
Memory bottleneck
Low traffic
Inefficient pipeline
```

High utilization isn't automatically perfect either.

You still need to monitor:

```text
Latency
Memory
Errors
Throughput
Quality
```

---

# 86. What Is AI Observability?

A production AI system should be measurable.

Useful metrics include:

```text
Request count
TTFT
End-to-end latency
Tokens/sec
Input tokens
Output tokens
GPU utilization
GPU memory
RAM
Queue length
Errors
Cost
Retrieval quality
Answer quality
```

AI engineering combines:

```text
Software engineering
+
ML
+
Infrastructure
+
Observability
```

---

# 87. A Complete Inference Request

Imagine the user asks:

```text
"Explain our company's Kubernetes deployment process."
```

A production application might do:

```text
User
 ↓
Authentication
 ↓
Application
 ↓
Query construction
 ↓
Retriever
 ↓
Relevant documents
 ↓
Reranker
 ↓
Prompt construction
 ↓
LLM
 ↓
Prefill
 ↓
Decode
 ↓
Response validation
 ↓
User
```

This is much closer to real enterprise AI engineering than simply:

```text
Prompt → LLM
```

---

# 88. Cost of Inference

For an API-based system, cost can depend on:

```text
Input tokens
+
Output tokens
+
Model
```

For self-hosted inference, cost includes:

```text
GPU/server
Electricity
Storage
Networking
Maintenance
Engineering
```

Therefore AI engineers optimize:

```text
Quality
+
Latency
+
Throughput
+
Cost
```

---

# 89. The 4-Way Tradeoff

When designing an LLM application, you often balance:

```text
          QUALITY
             /            /             /              /             COST ---- LATENCY
          \      /
           \    /
         THROUGHPUT
```

You usually cannot maximize everything simultaneously.

For example:

```text
Huge model
→ potentially better quality
→ more expensive
→ potentially slower
```

A smaller model:

```text
Lower cost
Potentially lower latency
Potentially lower quality
```

But a strong RAG/tool architecture can sometimes compensate.

---

# 90. Bigger Model vs Better System

Imagine:

### System A

```text
Very large model
Poor prompt
Poor retrieval
No validation
Poor evaluation
```

### System B

```text
Smaller model
Good prompt
Good retrieval
Good tools
Good validation
Good evaluation
```

System B can sometimes outperform System A for a specific business task.

Therefore:

> **AI engineering is not simply choosing the biggest model.**

---

# 91. What Is Model Distillation?

Knowledge distillation is a technique where a larger model can help train a smaller model.

Conceptually:

```text
Large teacher model
        ↓
Training signals
        ↓
Small student model
```

Goal:

```text
Smaller
+
Faster
+
Cheaper
```

while retaining useful capabilities.

---

# 92. What Is Speculative Decoding?

A simplified idea:

```text
Small fast model
      ↓
Proposes several tokens
      ↓
Large model
      ↓
Verifies/corrects
```

This can sometimes accelerate generation.

The key idea is:

> Use a fast model to propose likely tokens and a stronger model to verify them.

---

# 93. Day 5 Formula Sheet

## Weight memory

```text
Memory ≈ Parameters × Bytes per parameter
```

## FP32

```text
4 bytes/parameter
```

## FP16/BF16

```text
2 bytes/parameter
```

## INT8

```text
1 byte/parameter
```

## INT4

```text
0.5 byte/parameter
```

## Approximate generation time

```text
Output tokens / tokens-per-second
```

Example:

```text
300 tokens / 30 tokens/sec
≈ 10 sec
```

This ignores TTFT and other overhead.

---

# 94. Quick Comparison

| Model | FP16 weight estimate | INT8 estimate | INT4 estimate |
|---|---:|---:|---:|
| 3B | ~6 GB | ~3 GB | ~1.5 GB |
| 7B | ~14 GB | ~7 GB | ~3.5 GB |
| 13B | ~26 GB | ~13 GB | ~6.5 GB |
| 70B | ~140 GB | ~70 GB | ~35 GB |

These are **rough weight-only estimates**, not total runtime memory.

---

# 95. Beginner Interview Questions

## Q1. What does 7B mean?

Approximately 7 billion parameters.

## Q2. What is a parameter?

A numerical value learned by the model during training.

## Q3. What is FP16?

A 16-bit floating-point numerical format.

## Q4. What is INT4?

A 4-bit integer representation commonly used for quantized inference.

## Q5. What is quantization?

Representing model values using fewer bits to reduce memory/storage and potentially improve inference efficiency.

## Q6. What is VRAM?

Memory associated with a GPU.

## Q7. Why are GPUs useful for LLMs?

They efficiently perform large amounts of parallel tensor/matrix computation.

## Q8. What is KV cache?

Cached key/value attention information used during autoregressive generation to avoid repeating certain calculations.

## Q9. What is TTFT?

Time to first token.

## Q10. What is throughput?

The amount of work a system handles over time.

## Q11. What is prefill?

Processing the existing prompt/context before generating output.

## Q12. What is decode?

Generating output tokens after prefill.

## Q13. Why does context length matter?

Longer context can increase input-processing work and KV-cache memory requirements.

## Q14. Why is weight memory not total memory?

Because inference also requires KV cache, activations, buffers, runtime overhead and other memory.

---

# 96. Practical Exercise 1

Calculate approximate weight memory.

### 7B FP16

```text
7 × 2
=
14 GB
```

### 7B INT8

```text
7 × 1
=
7 GB
```

### 7B INT4

```text
7 × 0.5
=
3.5 GB
```

---

# 97. Practical Exercise 2

Calculate:

```text
13B FP16
```

Answer:

```text
13 × 2
=
26 GB
```

INT4:

```text
13 × 0.5
=
6.5 GB
```

---

# 98. Practical Exercise 3

Calculate:

```text
70B FP16
```

Answer:

```text
70 × 2
=
140 GB
```

INT4:

```text
70 × 0.5
=
35 GB
```

---

# 99. Practical Exercise 4

You have:

```text
8 GB GPU VRAM
```

You want:

```text
7B FP16
```

Approximate weights:

```text
14 GB
```

Can all weights fit?

```text
No.
```

Possible solutions:

```text
Use 4-bit quantization
Use smaller model
Use offloading
Use multiple GPUs
Use a GPU with more VRAM
```

---

# 100. Practical Exercise 5

A model generates:

```text
25 tokens/sec
```

and the answer contains:

```text
250 tokens
```

Approximate decode time:

```text
250 / 25
=
10 seconds
```

Actual end-to-end latency will also include:

```text
TTFT
Network
Scheduling
Processing overhead
```

---

# 101. Practical Exercise 6

RAG retrieves:

```text
2,000 tokens
```

User question:

```text
20 tokens
```

Before considering system instructions and history, the model needs to process approximately:

```text
2,020 tokens
```

Question:

> Does RAG affect inference performance?

Yes.

It increases input context and therefore affects prefill work, token usage and potentially memory.

---

# 102. The 15 Things You Must Remember

```text
1. 7B means approximately 7 billion parameters.

2. Parameter count affects weight memory.

3. FP32 ≈ 4 bytes/parameter.

4. FP16/BF16 ≈ 2 bytes/parameter.

5. INT8 ≈ 1 byte/parameter.

6. INT4 ≈ 0.5 byte/parameter.

7. Weight memory is not total inference memory.

8. VRAM is GPU-associated memory.

9. GPUs are excellent for parallel tensor operations.

10. Quantization reduces numerical precision and memory.

11. KV cache improves autoregressive generation efficiency but consumes memory.

12. TTFT means time to first token.

13. Tokens/sec measures generation speed.

14. Prefill processes the input context; decode generates output.

15. Production AI requires optimizing quality, latency, throughput, cost and reliability.
```

---

# 103. Full Day 4 + Day 5 Mental Model

## Training

```text
DATA
 ↓
TOKENS
 ↓
MODEL
 ↓
PREDICTION
 ↓
LOSS
 ↓
BACKPROPAGATION
 ↓
GRADIENTS
 ↓
OPTIMIZER
 ↓
PARAMETER UPDATE
 ↓
REPEAT
 ↓
TRAINED MODEL
```

## Inference

```text
USER PROMPT
 ↓
TOKENIZATION
 ↓
MODEL
 ↓
PREFILL
 ↓
KV CACHE
 ↓
DECODE
 ↓
NEXT TOKEN
 ↓
NEXT TOKEN
 ↓
NEXT TOKEN
 ↓
REPEAT
 ↓
RESPONSE
```

## Enterprise AI

```text
USER
 ↓
APPLICATION
 ↓
RETRIEVAL / TOOLS / MEMORY
 ↓
CONTEXT
 ↓
LLM
 ↓
VALIDATION
 ↓
APPLICATION
 ↓
USER
```

---

# 104. What You Should Now Be Able to Explain

You should now be comfortable explaining:

### Model

```text
What is a parameter?
What does 7B mean?
Why does model size matter?
```

### Numerical representation

```text
FP32
FP16
BF16
INT8
INT4
```

### Hardware

```text
CPU
GPU
RAM
VRAM
Unified memory
Tensor operations
```

### Inference

```text
Model loading
Prefill
Decode
TTFT
Tokens/sec
Latency
Throughput
```

### Optimization

```text
Quantization
KV cache
Batching
Offloading
```

### Production

```text
Model serving
Autoscaling
Rate limiting
Observability
Cost optimization
```

---

# 105. Final Interview Answer

If someone asks:

> **"How do you estimate whether an LLM can run on a machine?"**

A strong fresher-level answer is:

> First I would check the model's parameter count and numerical precision. A rough estimate for weight memory is the number of parameters multiplied by bytes per parameter. For example, a 7B model in FP16 needs roughly 14 GB for the weights, while a 4-bit version needs roughly 3.5 GB. However, total inference memory is higher because the system also needs memory for KV cache, activations, runtime buffers and other overhead. I would then compare this with available GPU VRAM or unified/system memory, consider context length and concurrency, and verify that the inference runtime supports the model and hardware.

---

# 106. Final Day 5 Takeaway

Day 4 answered:

> **How does an LLM learn?**

```text
Prediction
 ↓
Loss
 ↓
Gradient
 ↓
Optimizer
 ↓
Parameter update
```

Day 5 answered:

> **How do we run the trained LLM?**

```text
Model weights
 ↓
Memory
 ↓
Hardware
 ↓
Inference runtime
 ↓
Prompt
 ↓
Prefill
 ↓
KV cache
 ↓
Decode
 ↓
Tokens
 ↓
Response
```

The most important engineering mindset is:

```text
MODEL
  +
MEMORY
  +
COMPUTE
  +
RUNTIME
  +
CONTEXT
  +
OPTIMIZATION
  +
APPLICATION ARCHITECTURE
  =
PRODUCTION AI SYSTEM
```

---

# Day 6 Preview

Next we will open the LLM itself and understand what is actually happening inside:

```text
Text
 ↓
Tokens
 ↓
Token IDs
 ↓
Embeddings
 ↓
Positional information
 ↓
Query / Key / Value
 ↓
Self-Attention
 ↓
Multi-Head Attention
 ↓
Feed-Forward Network
 ↓
Residual Connections
 ↓
Layer Normalization
 ↓
Transformer Block
 ↓
Many Transformer Blocks
 ↓
Logits
 ↓
Next-token prediction
```

The main goal of **Day 6** will be to make:

> **"Attention is all you need"**

actually understandable to a fresher.
