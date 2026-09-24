# Day 3 — Transformers & Attention
## Beginner / Fresher Level

> **Goal:** Understand what happens inside an LLM between **input tokens** and the **next token it generates**.

You do **not** need advanced mathematics for this lesson. Focus on the mental model and the flow of information.

---

# 1. Where are we after Day 1 and Day 2?

## Day 1 — LLM generation

We learned that an LLM roughly follows:

```text
Text
  ↓
Tokens
  ↓
Transformer
  ↓
Logits
  ↓
Probabilities
  ↓
Temperature / Top-P
  ↓
Next Token
```

## Day 2 — Embeddings and retrieval

We learned:

```text
Text
  ↓
Embedding Model
  ↓
Vector
  ↓
Similarity Search
  ↓
Relevant Information
  ↓
RAG / LLM Context
```

## Day 3 — What happens inside the Transformer?

Today we focus on this part:

```text
Tokens
   ↓
Embeddings
   ↓
Positional Information
   ↓
Self-Attention
   ↓
Feed-Forward Network
   ↓
More Transformer Layers
   ↓
Logits
   ↓
Next Token
```

The most important concept today is:

# Attention

---

# 2. Why do we need attention?

Consider this sentence:

> **The animal didn't cross the road because it was tired.**

What does **"it"** refer to?

A human usually understands:

```text
it → animal
```

Now consider:

> **The animal didn't cross the road because the road was blocked.**

Now the important relationship is:

```text
blocked → road
```

The model has to figure out which words are related.

Attention helps the model ask:

> **Which other tokens are important for understanding this token?**

---

# 3. Attention using a simple highlighting example

Sentence:

> **The cat drank the milk because it was hungry.**

Look at the word:

> **it**

A useful mental model is that the model may pay different amounts of attention to different words.

Conceptually:

```text
The    cat    drank    the    milk    because    it    was    hungry
       ↑                           ↑              ↑             ↑
       └────────────── relevant information ────────────────────┘
```

The exact attention values are calculated mathematically by the model.

The key idea is simply:

> **Attention helps determine which tokens are relevant to another token.**

---

# 4. Self-attention

Why is it called **self-attention**?

Because the tokens in the same sequence attend to other tokens in that sequence.

Example:

```text
The dog chased the ball.
```

When processing `dog`, the model can consider:

```text
The
chased
ball
```

When processing `ball`, it can also consider:

```text
The
dog
chased
```

So:

> **Self-attention = tokens in the same sequence exchange information with one another.**

---

# 5. Example: Pronoun understanding

Consider:

> **The girl dropped the glass because she was scared.**

The model needs to connect:

```text
she → girl
```

Not:

```text
she → glass
```

Attention provides a mechanism for learning these relationships.

Another example:

> **The manager spoke to the engineer because he had the answer.**

Now the model has to use context to understand what `he` is referring to.

The point is not that one attention head literally stores "he = engineer". Rather, attention lets the network create representations that incorporate relationships among the tokens.

---

# 6. Attention is about relationships

You can think of a sentence as a group of connected tokens.

Example:

```text
The ── cat ── chased ── the ── mouse
          │               │
          └───────────────┘
             relationships
```

The model doesn't treat every token as completely isolated.

It builds context by allowing information to flow between tokens.

---

# 7. Query, Key and Value (Q, K, V)

This sounds complicated, but the basic idea is simple.

A token creates three different representations:

- **Query (Q)** → What am I looking for?
- **Key (K)** → What kind of information do I contain?
- **Value (V)** → What information should I provide?

A useful analogy is a library.

Imagine you ask:

> "I want books about machine learning."

Your request is the:

```text
Query
```

Each book has searchable information about what it contains:

```text
Key
```

Once a book is selected as relevant, you actually use its content:

```text
Value
```

So:

```text
Query → What am I looking for?
Key   → What do I represent?
Value → What information can I provide?
```

---

# 8. Q, K, V with a sentence

Sentence:

> **The cat drank milk.**

Suppose we are looking at the token:

```text
drank
```

The model can conceptually ask:

> "Which other tokens are useful for understanding `drank`?"

It compares the query from `drank` with keys from other tokens.

Conceptually:

```text
                   drank
                     │
                   Query
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
         The        cat        milk
          │          │          │
         Key        Key        Key
          │          │          │
          └──────────┼──────────┘
                     ↓
              relevance scores
                     ↓
              weighted Values
                     ↓
             updated representation
```

This is the core idea behind attention.

---

# 9. Attention scores

The model calculates how relevant each token is.

For example, imagine:

```text
Sentence:
The cat drank milk
```

For the word `drank`, suppose the model produces these illustrative attention weights:

```text
The      0.05
cat      0.40
drank    0.15
milk     0.40
```

This means, in this simplified example:

```text
cat  → highly relevant
milk → highly relevant
The  → less relevant
```

These numbers are only for intuition. Real models use learned transformations and matrix operations.

---

# 10. How attention is calculated — simple view

The process is roughly:

```text
Query
  ↓
Compare with Keys
  ↓
Relevance Scores
  ↓
Softmax
  ↓
Attention Weights
  ↓
Weighted combination of Values
  ↓
New representation
```

You may see the famous equation:

```text
Attention(Q, K, V)
= softmax(QKᵀ / √dₖ)V
```

At fresher level, don't try to memorize the formula yet.

Understand the flow:

```text
Q + K
  ↓
How relevant?
  ↓
Weights
  ↓
Use V
```

---

# 11. Why is softmax involved?

We already saw softmax on Day 1.

Softmax converts raw scores into a normalized distribution.

For example:

```text
Raw scores:

cat      4.2
milk     4.0
The      1.1
```

After softmax, you might get something like:

```text
cat      0.49
milk     0.44
The      0.07
```

Now the numbers can be interpreted as relative attention weights.

The sum is approximately:

```text
1.00
```

---

# 12. Why do we need positional information?

Here's a very important problem.

Compare:

```text
Dog bites man.
```

and:

```text
Man bites dog.
```

The same words appear, but the meaning is very different.

The model therefore needs information about **position/order**.

Conceptually:

```text
The      dog      bites      man
Pos 1    Pos 2    Pos 3      Pos 4
```

Without some representation of order, the model would have much less information about sequence structure.

---

# 13. Positional information in modern Transformers

Older Transformer descriptions often show a fixed positional encoding.

Modern models can use other techniques.

A very important example is:

> **RoPE — Rotary Positional Embeddings**

You don't need the mathematics today.

At fresher level, remember:

> **Positional information helps the model understand where tokens occur and how token order matters.**

---

# 14. Example: Why position matters

Consider:

```text
"Python is easy to learn."
```

and:

```text
"Easy to learn is Python."
```

The same words are present, but the structure is different.

Order matters for language.

Positional information helps the Transformer account for that order.

---

# 15. What is a Transformer block?

A simplified Transformer block looks like:

```text
Input
  ↓
Self-Attention
  ↓
Add & Normalize
  ↓
Feed-Forward Network
  ↓
Add & Normalize
  ↓
Output
```

Modern implementations vary in exact details and ordering, but this is a very useful beginner mental model.

---

# 16. What is the Feed-Forward Network?

Attention answers something like:

> **Which information should I pay attention to?**

The feed-forward network then transforms the information at each token position.

A simple analogy:

```text
Attention
→ gather useful information

Feed-forward network
→ process and transform that information
```

So:

```text
Attention:
"Who/what is relevant?"

Feed-Forward:
"What should I do with the information?"
```

This is a simplified intuition, not a complete mathematical definition.

---

# 17. Why do we need many Transformer layers?

One Transformer block is not enough for a capable LLM.

The model stacks many blocks.

Conceptually:

```text
Input
 ↓
Layer 1
 ↓
Layer 2
 ↓
Layer 3
 ↓
Layer 4
 ↓
...
 ↓
Layer N
 ↓
Output representation
```

Each layer further transforms the representations.

A useful intuition is:

```text
Early processing
→ local/basic patterns

More layers
→ richer contextual relationships

Later processing
→ increasingly useful high-level representations
```

Do not assume that each specific layer has one human-defined job such as "grammar layer" or "meaning layer". The model learns distributed representations across the network.

---

# 18. Why multiple attention heads?

Language contains many kinds of relationships.

For example, while processing a sentence, useful relationships could include:

```text
subject ↔ verb
verb ↔ object
pronoun ↔ noun
word ↔ nearby context
word ↔ distant context
```

Instead of using only one attention mechanism, Transformers use:

> **Multi-Head Attention**

Conceptually:

```text
                  Input
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
      Head 1      Head 2      Head 3
        │           │           │
        └───────────┼───────────┘
                    ↓
                  Combine
                    ↓
                Next layer
```

Each head gets an opportunity to learn different attention patterns.

Important:

> Head 1 is not guaranteed to mean "grammar", Head 2 is not guaranteed to mean "pronouns", etc.

The model learns useful patterns during training.

---

# 19. Multi-head attention with an example

Sentence:

> **The bank approved the loan near the river.**

The word:

```text
bank
```

is ambiguous.

It could be:

```text
financial institution
```

or:

```text
river bank
```

Different attention patterns can capture different relationships.

For example:

```text
bank → approved → loan
```

suggests financial meaning.

While:

```text
bank → river
```

suggests geographic meaning.

This is only an intuition for why multiple attention patterns can be useful.

---

# 20. Encoder vs Decoder

The original Transformer architecture introduced:

```text
Encoder → Decoder
```

But modern LLM architectures differ.

## Encoder

Primarily turns input text into rich representations.

Conceptually:

```text
Text
 ↓
Encoder
 ↓
Representation
```

Encoder-style models are useful for tasks such as:

- classification
- representation learning
- retrieval
- some embedding use cases

A famous example is the BERT family.

## Decoder

Primarily generates text autoregressively.

```text
Context
 ↓
Decoder
 ↓
Next token
 ↓
Next token
 ↓
Next token
```

GPT-style models are examples of **decoder-only Transformers**.

---

# 21. Why decoder-only works well for generation

Suppose the model receives:

```text
The capital of France is
```

It predicts:

```text
Paris
```

Now the sequence is:

```text
The capital of France is Paris
```

Then it predicts the next token.

Maybe:

```text
.
```

Then it repeats.

The generation loop is:

```text
Context
   ↓
Predict next token
   ↓
Append token
   ↓
Predict next token
   ↓
Append token
   ↓
Repeat
```

This is called **autoregressive generation**.

---

# 22. Causal attention

There is an important restriction during autoregressive generation:

> The model should not look at future tokens that have not yet been generated.

Suppose:

```text
The cat is sleeping
```

When predicting `sleeping`, the model can use:

```text
The
cat
is
```

but not a future token after `sleeping`.

Conceptually:

```text
             The   cat   is   sleeping

The           ✓     ✗    ✗       ✗
cat           ✓     ✓    ✗       ✗
is            ✓     ✓    ✓       ✗
sleeping      ✓     ✓    ✓       ✓
```

This is called:

> **Causal masking**

The exact implementation can be optimized heavily in real systems, but the core idea is simple:

```text
No peeking into the future.
```

---

# 23. One complete example

Let's trace:

> **The dog chased the ball because it was excited.**

## Step 1 — Tokenization

Conceptually:

```text
The | dog | chased | the | ball | because | it | was | excited | .
```

The actual tokenizer may split some words differently.

---

## Step 2 — Embeddings

Each token becomes a vector representation.

```text
The      → vector
dog      → vector
chased   → vector
ball     → vector
...
```

---

## Step 3 — Positional information

The model also gets information related to token positions.

```text
The      → position 1
dog      → position 2
chased   → position 3
...
```

---

## Step 4 — Self-attention

The token `it` can build a representation influenced by relevant tokens such as:

```text
dog
ball
excited
```

The model is not simply performing a rule-based lookup. It is combining learned representations using attention.

---

## Step 5 — Feed-forward processing

The representations are transformed further.

---

## Step 6 — More Transformer blocks

This process is repeated through many blocks.

```text
Block 1
 ↓
Block 2
 ↓
Block 3
 ↓
...
 ↓
Block N
```

---

## Step 7 — Final logits

The model produces scores for possible next tokens.

Conceptually:

```text
Token        Score
-------------------
.             8.1
because       3.2
and           2.8
the           2.1
...
```

---

## Step 8 — Probabilities

Softmax converts the scores into probabilities.

---

## Step 9 — Sampling

Temperature and Top-P can influence how the next token is selected.

---

## Step 10 — Output token

A token is selected.

Then the process repeats.

---

# 24. The complete LLM pipeline

Now combine everything from Days 1–3.

```text
                         TEXT
                           │
                           ▼
                      TOKENIZER
                           │
                           ▼
                     TOKEN IDs
                           │
                           ▼
                     EMBEDDINGS
                           │
                           ▼
              POSITIONAL INFORMATION
                           │
                           ▼
                ┌─────────────────────┐
                │     TRANSFORMER     │
                │                     │
                │   Self-Attention    │
                │         ↓           │
                │   Feed-Forward      │
                │         ↓           │
                │   Normalization     │
                └──────────┬──────────┘
                           │
                       repeat many
                          times
                           │
                           ▼
                         LOGITS
                           │
                           ▼
                     PROBABILITIES
                           │
                           ▼
                TEMPERATURE / TOP-P
                           │
                           ▼
                      NEXT TOKEN
                           │
                           └───────► repeat
```

This is the mental model you should keep.

---

# 25. Another analogy: a team discussion

Imagine 10 people in a meeting:

```text
Developer
Security Engineer
DBA
DevOps
Architect
Product Manager
...
```

You ask:

> "Why did the deployment fail?"

You don't give equal attention to everyone.

You naturally pay more attention to:

```text
DevOps
Developer
Security Engineer
```

because their information is likely to be relevant.

Now imagine that every person also has:

```text
Query → What information am I looking for?
Key   → What information do I specialize in?
Value → What information can I contribute?
```

That gives you a useful intuition for Q/K/V attention.

---

# 26. Attention does NOT mean the model "thinks like a human"

This is an important engineering mindset.

Attention is a mathematical mechanism that lets the model combine information across tokens.

It does **not** mean:

```text
The model literally looks at words with human eyes.
```

And it does not automatically mean:

```text
One attention head = one specific human concept.
```

Instead:

> The model learns patterns in numerical representations.

---

# 27. Common fresher misconceptions

## Misconception 1:
### "Attention tells the model which word is important."

Partly true as an intuition, but oversimplified.

More precisely:

> Attention computes weighted interactions between token representations.

---

## Misconception 2:
### "Embedding = token"

No.

```text
Token
→ unit of text representation

Embedding
→ vector representation
```

---

## Misconception 3:
### "Every attention head has one fixed meaning."

No.

Heads learn different patterns, but those patterns are not guaranteed to have simple human-readable labels.

---

## Misconception 4:
### "The Transformer is only attention."

No.

A Transformer block also contains feed-forward processing, residual connections, normalization, and other implementation details.

---

## Misconception 5:
### "LLM reads the whole answer before writing it."

For autoregressive generation, the model predicts tokens progressively.

Conceptually:

```text
token 1
  ↓
token 2
  ↓
token 3
  ↓
...
```

---

# 28. What is a residual connection?

You will see this in Transformer diagrams:

```text
Input ────────────────┐
  │                   │
  ▼                   │
Attention             │
  │                   │
  └─────── Add ◄──────┘
```

The idea is to preserve and reuse information from an earlier representation while adding the newly transformed information.

This helps deep neural networks train effectively.

At fresher level:

> **Residual connection = add the original signal back to the transformed signal.**

You do not need the mathematics today.

---

# 29. What is normalization?

Transformers also use normalization around their sublayers.

Very simplified:

```text
Representation
      ↓
Normalize
      ↓
More stable numerical processing
```

The exact choice and placement can differ by architecture.

For fresher understanding:

> **Normalization helps keep the internal values in a useful numerical range and supports stable training.**

---

# 30. Day 3 in one picture

```text
                    INPUT TEXT
                        │
                        ▼
                    TOKENIZER
                        │
                        ▼
                     TOKENS
                        │
                        ▼
                   EMBEDDINGS
                        │
                        ▼
                 POSITIONAL INFO
                        │
                        ▼
        ┌─────────────────────────────┐
        │      TRANSFORMER BLOCK       │
        │                             │
        │  Self-Attention             │
        │       │                     │
        │       ▼                     │
        │  Information Mixing         │
        │       │                     │
        │       ▼                     │
        │  Feed-Forward               │
        │       │                     │
        │       ▼                     │
        │  Add / Normalize            │
        └──────────────┬──────────────┘
                       │
                repeated many times
                       │
                       ▼
                     LOGITS
                       │
                       ▼
                  PROBABILITIES
                       │
                       ▼
                SAMPLING CONTROLS
                       │
                       ▼
                   NEXT TOKEN
```

---

# 31. The 10 concepts you should know after Day 3

### 1. Token
A unit of text used by the model.

### 2. Embedding
A vector representation of an input.

### 3. Positional information
Information that helps represent order.

### 4. Self-attention
A mechanism that allows token representations to interact with each other.

### 5. Query
"What am I looking for?"

### 6. Key
"What kind of information do I represent?"

### 7. Value
"What information should I provide?"

### 8. Multi-head attention
Multiple attention mechanisms operating in parallel and then combined.

### 9. Transformer block
A repeated building block containing attention, feed-forward processing, normalization, residual connections, and related components.

### 10. Causal masking
Prevents an autoregressive decoder from using future tokens during generation.

---

# 32. Day 1 + Day 2 + Day 3

At this point, you should be able to see the complete picture.

## Generation

```text
User Prompt
    ↓
Tokens
    ↓
Embeddings + Position
    ↓
Transformer
    ↓
Attention
    ↓
Feed-Forward
    ↓
More Transformer Blocks
    ↓
Logits
    ↓
Probability Distribution
    ↓
Temperature / Top-P
    ↓
Next Token
```

## Retrieval / RAG

```text
Documents
    ↓
Chunking
    ↓
Embedding Model
    ↓
Vectors
    ↓
Vector Database
    ↓
Similarity Search
    ↓
Relevant Chunks
    ↓
LLM Context
    ↓
Transformer
    ↓
Answer
```

---

# 33. Fresher-level interview questions

Try answering these without looking back.

### Q1. What is a Transformer?

A neural network architecture that processes token representations using mechanisms such as self-attention and feed-forward networks.

### Q2. Why do we need attention?

To allow information from different tokens to interact and help the model represent relationships and context.

### Q3. What are Q, K and V?

```text
Q → query
K → key
V → value
```

They are learned projections used to calculate and apply attention.

### Q4. Why is positional information needed?

Because token order affects meaning.

### Q5. What is multi-head attention?

Multiple attention mechanisms operating in parallel so the model can learn different interaction patterns.

### Q6. What is causal masking?

A restriction that prevents a decoder from attending to future tokens during autoregressive generation.

### Q7. Why do we stack Transformer blocks?

To progressively transform and enrich the token representations.

### Q8. What is the difference between attention and feed-forward processing?

A useful intuition is:

```text
Attention → mix information across tokens
Feed-forward → transform the representation at each position
```

### Q9. What is a decoder-only Transformer?

A Transformer architecture designed primarily for autoregressive text generation.

### Q10. Where does temperature act?

During token selection/sampling from the model's output distribution.

---

# 34. A simple coding mental model

You can mentally represent a simplified LLM call as:

```python
tokens = tokenize("The cat drank milk")

x = embed(tokens)

x = add_position_information(x)

for block in transformer_blocks:
    x = self_attention(x)
    x = feed_forward(x)

logits = output_layer(x)

probabilities = softmax(logits)

next_token = sample(probabilities)
```

This is **not production Transformer code**. It is a conceptual model of the pipeline.

---

# 35. What you do NOT need to master today

Do not get stuck on:

- Matrix multiplication details
- Eigenvectors
- Backpropagation math
- Attention optimization kernels
- FlashAttention
- KV-cache implementation details
- Quantization mathematics
- GPU kernel programming

Those come later.

For now, get this flow firmly in your head:

```text
Token
 ↓
Embedding
 ↓
Position
 ↓
Attention
 ↓
Feed-forward
 ↓
More layers
 ↓
Logits
 ↓
Next token
```

---

# Day 3 checkpoint

You are ready to move on when you can explain these in simple language:

1. Why does the model need attention?
2. What is self-attention?
3. What are Query, Key and Value?
4. Why are embeddings needed?
5. Why does token order matter?
6. What is positional information?
7. Why are there multiple attention heads?
8. What does a feed-forward network do?
9. Why are Transformer blocks stacked?
10. What is causal masking?
11. What is a decoder-only Transformer?
12. Where do logits come from?
13. Where do temperature and Top-P fit?
14. How does this connect to RAG?

---

# The one diagram to memorize

```text
                    USER INPUT
                        │
                        ▼
                     TOKENS
                        │
                        ▼
                    EMBEDDINGS
                        │
                        ▼
               POSITIONAL INFORMATION
                        │
                        ▼
                 SELF-ATTENTION
                        │
                        ▼
                FEED-FORWARD NETWORK
                        │
                        ▼
                MORE TRANSFORMER LAYERS
                        │
                        ▼
                     LOGITS
                        │
                        ▼
                  PROBABILITIES
                        │
                        ▼
                TEMPERATURE / TOP-P
                        │
                        ▼
                   NEXT TOKEN
                        │
                        └───────► repeat
```

> **Core idea:** A Transformer repeatedly transforms token representations, using attention to mix information across tokens, until the model can produce scores for the next token.
