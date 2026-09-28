# Day 3 — What Happens Inside an LLM?
## Transformers & Attention — Absolute Fresher Level

### Goal

Understand, from zero, how an LLM goes from:

```text
Your text
  ↓
Tokens
  ↓
Numbers / embeddings
  ↓
Transformer
  ↓
Attention
  ↓
Next-token probabilities
  ↓
Next token
```

---

## 1. Start with a simple example

You type:

> The cat is sitting on the

The model needs to predict:

```text
?
```

Possible next tokens could be:

```text
mat
chair
floor
bed
...
```

The model calculates which possibilities are most appropriate.

---

# 2. Step 1 — Text becomes tokens

A tokenizer breaks text into pieces called **tokens**.

For example:

```text
I love coffee.
```

Conceptually:

```text
I | love | coffee | .
```

Important:

> A token is not always a complete word.

A long or uncommon word can be split into multiple tokens.

```text
Text
 ↓
Tokenizer
 ↓
Tokens
```

---

# 3. Step 2 — Tokens become numbers

Neural networks operate on numbers.

A tokenizer assigns an ID to each token.

Illustrative example:

```text
The   → 101
cat   → 532
sat   → 781
mat   → 921
```

These are **token IDs**.

But token IDs don't directly contain enough semantic information.

For example:

```text
cat → 532
dog → 711
car → 982
```

The numbers don't mean that `cat` and `dog` are related.

So we need embeddings.

---

# 4. Step 3 — Embeddings

An **embedding** is a vector representation.

For a beginner, think:

> A vector is simply a list of numbers that represents learned information.

Illustratively:

```text
cat → [0.2, 0.8, -0.1, 0.5, ...]
dog → [0.3, 0.7, -0.2, 0.6, ...]
car → [0.9, -0.4, 0.7, -0.2, ...]
```

Real models use much larger vectors.

A useful intuition is:

```text
Related concepts
      ↓
often have related representations
```

---

# 5. Simple embedding analogy

Imagine every concept gets a location on a giant map.

```text
             ANIMALS

        dog       cat
          \      /
           \    /
            lion


                         VEHICLES

                       car
                      truck
```

Words with related meanings tend to have related locations in the learned representation space.

This is the intuition behind embeddings.

---

# 6. Why embeddings alone aren't enough

Consider:

> The bank approved my loan.

and:

> I sat near the bank of the river.

The word `bank` appears in both sentences, but it means something different.

So the model needs to understand:

> What does this token mean in this particular context?

This is where **attention** becomes important.

---

# 7. Attention — the simplest explanation

Attention helps the model answer:

> **Which other tokens contain information that is relevant to this token?**

Example:

> The cat drank milk.

Look at:

```text
drank
```

Useful information includes:

```text
cat  → who drank?
milk → what was drunk?
```

Conceptually:

```text
        cat
         ↑
         │
       drank
         │
         ↓
        milk
```

Attention lets information from other tokens influence the representation being built.

---

# 8. Attention is like looking around

Imagine a classroom.

The teacher asks:

> Who submitted the assignment?

You naturally pay attention to:

```text
students
teacher
assignment
```

You probably don't need much information from:

```text
wall
fan
window
```

Attention is somewhat like this:

> The model calculates which information is relevant to what it is processing.

It is a mathematical mechanism, not human-like looking.

---

# 9. Self-attention

**Self-attention** means that tokens in the same sequence can interact with other tokens in that sequence.

Example:

```text
The | dog | chased | the | ball
```

The representation of `dog` can be influenced by other tokens.

The representation of `ball` can also be influenced by other tokens.

So:

```text
Tokens
  ↓
Look at other tokens
  ↓
Combine relevant information
  ↓
Create better contextual representations
```

---

# 10. Why is self-attention useful?

Consider:

> The boy put the book on the table because it was heavy.

What does `it` probably refer to?

```text
it → book
```

The model needs to connect information across the sentence.

Attention provides a mechanism for doing this.

---

# 11. Long-distance relationships

Consider:

> The software engineer who joined the company last year fixed the production bug.

A useful relationship is:

```text
engineer → fixed → bug
```

The words are separated by several other tokens.

Attention allows information to flow between token representations across the sequence.

---

# 12. Important: attention is not human thinking

Do not imagine:

```text
Model sees word
    ↓
Human-like thought
    ↓
Understanding
```

Instead:

```text
Numerical representations
        ↓
Mathematical operations
        ↓
Relationships between representations
        ↓
New representations
```

Attention is one of those mathematical mechanisms.

---

# 13. Query, Key and Value

You will hear:

```text
Q = Query
K = Key
V = Value
```

The easiest analogy is a library.

---

## Query

You ask:

> I want books about machine learning.

This is the:

**Query**

Think:

> **What information am I looking for?**

---

## Key

Each book has information about what it contains:

```text
Book A → Cooking
Book B → Machine Learning
Book C → History
```

Those searchable characteristics are like:

**Keys**

Think:

> **What kind of information do I represent?**

---

## Value

Once you find the relevant book, you actually use its content.

That is the:

**Value**

Think:

> **What information can I provide?**

---

## Q/K/V in one table

| Concept | Simple question |
|---|---|
| Query | What am I looking for? |
| Key | What do you represent? |
| Value | What information can you provide? |

---

# 14. Q/K/V with a sentence

Sentence:

> The cat drank milk.

Suppose we're processing:

```text
drank
```

Conceptually:

```text
                 drank
                   |
                 Query
                   |
          +--------+--------+
          ↓        ↓        ↓
         The      cat      milk
          |        |        |
         Key      Key      Key
          |        |        |
          +--------+--------+
                   ↓
             relevance scores
                   ↓
                 Values
                   ↓
          updated representation
```

The model compares the query with keys and uses the resulting weights to combine values.

---

# 15. Attention scores

The model calculates how relevant other tokens are.

Illustrative example:

```text
Sentence:
The cat drank milk
```

For `drank`:

```text
The       → 0.05
cat       → 0.40
drank     → 0.15
milk      → 0.40
```

These numbers are only for intuition.

They mean, roughly:

```text
cat  → more relevant
milk → more relevant
The  → less relevant
```

---

# 16. What is positional information?

Compare:

```text
Dog bites man.
```

with:

```text
Man bites dog.
```

Same words.

Different meaning.

Therefore the model needs information about token order.

Conceptually:

```text
The    cat    drank    milk
 1      2       3       4
```

The model needs both:

```text
WHAT is the token?
```

and:

```text
WHERE is the token?
```

---

# 17. RoPE

You may hear:

> **RoPE — Rotary Positional Embeddings**

Don't worry about the mathematics yet.

At fresher level:

> Positional techniques help the model represent token order and relative position.

RoPE is one common technique used in modern Transformer architectures.

---

# 18. What is a Transformer?

A **Transformer** is a neural-network architecture used by modern LLMs.

A simplified Transformer block looks like:

```text
Input
  ↓
Attention
  ↓
Feed-forward processing
  ↓
Output
```

Real Transformer implementations contain additional components such as:

- residual connections
- normalization
- projections
- positional techniques
- other implementation optimizations

For now, focus on the big picture.

---

# 19. What does the Transformer do?

Suppose the input is:

> The dog chased the ball.

The model starts with token representations.

Attention allows the representations to interact.

Then other neural-network operations transform them.

After processing, the representation of `dog` can contain information influenced by:

```text
dog
+
chased
+
ball
+
surrounding context
```

This is why we talk about **contextual representations**.

---

# 20. What is a contextual representation?

Compare:

```text
bank
```

with:

```text
The bank approved my loan.
```

and:

```text
I sat near the bank of the river.
```

The same token appears in different contexts.

The surrounding tokens change the information available to the model.

So the model can produce different contextual representations.

---

# 21. Feed-forward network

A Transformer also contains feed-forward neural-network components.

A useful beginner intuition:

```text
Attention
↓
Mix information across tokens

Feed-forward network
↓
Further transform the information
```

So you can remember:

```text
Attention:
"What information is relevant?"

Feed-forward:
"How should the representation be transformed?"
```

This is a simplified mental model.

---

# 22. Why are there many Transformer layers?

One block isn't enough for a large language model.

So blocks are stacked:

```text
Input
 ↓
Block 1
 ↓
Block 2
 ↓
Block 3
 ↓
Block 4
 ↓
...
 ↓
Block N
 ↓
Output
```

Each block further transforms the representations.

Don't assume that each layer has one simple human-defined job.

The model learns distributed representations across the network.

---

# 23. Multiple attention heads

A Transformer doesn't use just one attention calculation.

It uses **multi-head attention**.

Conceptually:

```text
                 Input
                   |
        +----------+----------+
        ↓          ↓          ↓
      Head 1     Head 2     Head 3
        |          |          |
        +----------+----------+
                   ↓
                Combine
                   ↓
               Next layer
```

Different heads can learn different interaction patterns.

For example, useful relationships may include:

```text
subject ↔ verb
verb ↔ object
pronoun ↔ noun
word ↔ nearby context
word ↔ distant context
```

But don't assume:

```text
Head 1 = grammar
Head 2 = nouns
Head 3 = verbs
```

The learned behavior is more complicated.

---

# 24. Encoder vs Decoder

Two important Transformer terms are:

## Encoder

Think:

> **Process/represent input.**

```text
Input
 ↓
Encoder
 ↓
Representation
```

## Decoder

Think:

> **Generate output.**

```text
Context
 ↓
Decoder
 ↓
Next token
```

Many modern generative LLMs, including GPT-style architectures, are based primarily on **decoder-only Transformers**.

---

# 25. Autoregressive generation

Suppose you type:

> The capital of India is

The model predicts something like:

```text
New
```

Now the sequence becomes:

```text
The capital of India is New
```

Then it predicts:

```text
Delhi
```

Now:

```text
The capital of India is New Delhi
```

Then it predicts the next token.

This is called:

> **Autoregressive generation**

Simple meaning:

> **Use what has already been generated to help generate the next token.**

---

# 26. Causal masking

There is an important rule during autoregressive generation:

> **The model must not peek at future tokens.**

For example, when predicting:

```text
sleeping
```

from:

```text
The cat is
```

the model can use the previous context.

It cannot use a future answer that has not been generated yet.

This restriction is called:

**Causal masking**

Simple definition:

> **Don't let the model see the future.**

---

# 27. Why causal masking matters

Imagine an exam asking:

> What is the next word?

But the complete answer is printed underneath the question.

The student could simply copy it.

Similarly, the language model needs to learn next-token prediction without being given the future answer.

---

# 28. Complete example

Input:

> The cat is sitting on the

We want:

```text
?
```

### Step 1 — Tokenization

Conceptually:

```text
The | cat | is | sitting | on | the
```

### Step 2 — Token IDs

Each token gets an ID.

```text
The     → ID
cat     → ID
is      → ID
sitting → ID
on      → ID
the     → ID
```

### Step 3 — Embeddings

Each token ID maps to a learned vector.

```text
The     → vector
cat     → vector
is      → vector
sitting → vector
on      → vector
the     → vector
```

### Step 4 — Position

```text
The      → position 1
cat      → position 2
is       → position 3
sitting  → position 4
on       → position 5
the      → position 6
```

### Step 5 — Attention

The model creates contextual relationships.

For example:

```text
cat ↔ sitting
sitting ↔ on
on ↔ the
```

There are many other relationships too.

### Step 6 — Feed-forward processing

The representations are transformed further.

### Step 7 — More Transformer blocks

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

### Step 8 — Logits

The model produces raw scores for possible next tokens.

Illustratively:

```text
mat      → 8.2
floor    → 6.5
chair    → 5.9
bed      → 4.8
table    → 3.1
```

These are **logits**, not probabilities.

### Step 9 — Probabilities

They can be converted into probabilities:

```text
mat      → 55%
floor    → 20%
chair    → 12%
bed      → 8%
table    → 5%
```

These numbers are illustrative.

### Step 10 — Sampling

Temperature and Top-P can influence token selection.

The model might select:

```text
mat
```

### Step 11 — Repeat

Now the context becomes:

```text
The cat is sitting on the mat
```

The model predicts the next token again.

---

# 29. The complete LLM generation loop

```text
                USER PROMPT
                     |
                     v
                   TOKENS
                     |
                     v
                 EMBEDDINGS
                     |
                     v
            POSITION INFORMATION
                     |
                     v
                TRANSFORMER
                     |
             +-------+-------+
             |               |
             v               v
         ATTENTION      FEED-FORWARD
             |               |
             +-------+-------+
                     |
                     v
                MORE LAYERS
                     |
                     v
                   LOGITS
                     |
                     v
                PROBABILITIES
                     |
                     v
             TEMPERATURE/TOP-P
                     |
                     v
                NEXT TOKEN
                     |
                     v
              Add to context
                     |
                     +----------> repeat
```

---

# 30. Keep these concepts separate

## Embedding

Roughly:

> **What does this token represent?**

## Attention

Roughly:

> **What other information is relevant to this token?**

## Feed-forward network

> **Further transforms the representation.**

## Logits

> **Raw scores for possible output tokens.**

## Temperature

> **Influences how the probability distribution is sampled.**

---

# 31. Day 1 + Day 2 + Day 3

## Day 1

We learned:

```text
Tokens
Input
Output
Temperature
Top-P
Context Window
```

## Day 2

We learned:

```text
Embeddings
Vectors
Similarity
RAG
```

## Day 3

We connect them:

```text
                 INPUT
                   |
                   v
                 TOKENS
                   |
                   v
               EMBEDDINGS
                   |
                   v
              TRANSFORMER
                   |
          +--------+--------+
          v                 v
      ATTENTION       FEED-FORWARD
          |                 |
          +--------+--------+
                   |
                   v
             MORE LAYERS
                   |
                   v
                LOGITS
                   |
                   v
             PROBABILITIES
                   |
                   v
           TEMPERATURE/TOP-P
                   |
                   v
              NEXT TOKEN
```

---

# 32. Where does RAG fit?

Suppose you ask:

> What is our company's password policy?

The LLM may not know your private company policy.

RAG can retrieve the relevant document.

```text
Company Documents
       |
       v
   Chunking
       |
       v
   Embeddings
       |
       v
 Vector Database
       |
       v
Similarity Search
       |
       v
Relevant Chunks
       |
       v
      LLM
       |
       v
 Transformer
       |
       v
    Answer
```

Important:

> **RAG does not replace the Transformer.**

RAG provides additional information/context to the LLM.

---

# 33. Simple RAG analogy

Imagine an exam.

The LLM is the student.

RAG is giving the student the relevant pages from the textbook.

Without RAG:

```text
Question
   ↓
Student's existing knowledge
   ↓
Answer
```

With RAG:

```text
Question
   ↓
Find relevant textbook pages
   ↓
Give pages to student
   ↓
Student uses them
   ↓
Answer
```

---

# 34. Why attention matters in RAG

Suppose RAG retrieves:

```text
Password must contain 12 characters.
Passwords expire every 90 days.
MFA is required for production access.
```

The user asks:

> How long should my password be?

The model has all three facts in its context.

Attention helps it process relationships such as:

```text
password → 12 characters
password → 90 days
MFA → production access
```

The relevant information can influence the generated answer.

---

# 35. What you do NOT need to master today

Don't get stuck on:

- Matrix multiplication
- Eigenvectors
- Backpropagation mathematics
- FlashAttention
- KV-cache implementation
- Quantization mathematics
- GPU kernel programming

Those come later.

For now, understand:

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

# 36. Day 3 glossary

| Term | Simple meaning |
|---|---|
| Token | Piece of text processed by the model |
| Token ID | Number representing a token |
| Embedding | Vector representation of a token |
| Vector | List of numbers |
| Attention | Mechanism for mixing relevant information |
| Self-attention | Tokens attend to other tokens in the same sequence |
| Query | What information am I looking for? |
| Key | What information do I represent? |
| Value | What information can I provide? |
| Attention score | How relevant one token is to another |
| Attention head | One attention mechanism within multi-head attention |
| Multi-head attention | Multiple attention mechanisms operating together |
| Transformer | Neural-network architecture used by modern LLMs |
| Feed-forward network | Neural network that further transforms representations |
| Positional information | Information about token order |
| Causal masking | Prevents seeing future tokens |
| Decoder-only | Transformer architecture commonly used for text generation |
| Logits | Raw scores for possible output tokens |
| Sampling | Choosing the next token from the output distribution |

---

# 37. The 5 concepts you absolutely must understand

If you remember only five things, remember these.

## 1. Tokens

```text
Sentence
 ↓
Pieces of text
```

## 2. Embeddings

```text
Token
 ↓
Numbers representing learned information
```

## 3. Attention

```text
Token
 ↓
"What other information is relevant?"
```

## 4. Transformer

```text
Attention
+
Feed-forward processing
+
many repeated layers
```

## 5. Next-token prediction

```text
Context
 ↓
Transformer
 ↓
Possible next tokens
 ↓
Select one
 ↓
Repeat
```

---

# 38. Final mental model

If someone asks:

> **How does ChatGPT generate an answer?**

A good fresher-level answer is:

> "The input is broken into tokens. The tokens are converted into numerical representations called embeddings, along with information about their positions. A Transformer processes these representations using self-attention and feed-forward layers. Attention allows the model to combine information from different tokens and understand their relationships. After many Transformer layers, the model produces scores called logits for possible next tokens. These are converted into probabilities, and sampling controls such as temperature and Top-P influence which token is selected. The selected token is added to the context and the process repeats."

That is an excellent starting mental model.

---

# 39. Day 3 checkpoint

Try answering these **without looking back**:

1. What is a token?
2. Why do tokens need to become numbers?
3. What is an embedding?
4. What problem does attention solve?
5. What is self-attention?
6. What are Query, Key and Value?
7. Why does token position matter?
8. What is a Transformer?
9. What is a Transformer block?
10. What is multi-head attention?
11. What is a feed-forward network?
12. Why are many Transformer layers used?
13. What is a decoder-only Transformer?
14. What is causal masking?
15. What are logits?
16. Where does temperature come into the process?
17. How does RAG provide information to the LLM?

---

# 40. The one sentence to remember

> **An LLM converts text into numerical representations, uses Transformer layers and attention to combine contextual information, and repeatedly predicts the next token.**

---

# Day 4 Preview

Next we will answer:

> **How does an LLM actually learn all of this?**

We will start from absolute zero:

```text
Neuron
 ↓
Neural Network
 ↓
Weights
 ↓
Parameters
 ↓
Training Data
 ↓
Prediction
 ↓
Loss
 ↓
Gradient Descent
 ↓
Backpropagation
 ↓
Pretraining
 ↓
Fine-tuning
 ↓
Instruction Tuning
 ↓
RLHF / DPO
```

No advanced mathematics initially. We'll build each concept step by step.
