# ROLE: LEAD HFT SYSTEM ARCHITECT & HARDWARE OPTIMIZATION SPECIALIST

**Context:** You are building the core execution unit for "Project Event Horizon" — a high-frequency trading system running on C++23. The goal is not just low latency, but **predictable deterministic latency** under extreme contention.

## TASK: SPECULATIVE ZERO-COPY KERNEL WITH TRANSACTIONAL MEMORY SEMANTICS

Implement a single-file C++23 kernel that handles an event stream.
**The Twist:** The engine operates **speculatively**. It must reserve space and begin processing an order *before* full risk validation is complete. If validation fails, the operation must be **Rolled Back** in a Wait-Free manner without blocking consumers.

### STRICT TECHNICAL REQUIREMENTS:

#### 1. Transactional Lock-Free Ring Buffer (The Logic Trap)
Implement a Disruptor-style Ring Buffer, but with **Atomic Transaction Semantics**:
- **Two-Phase Commit:** A Producer reserves a slot and writes data (Phase 1). It then either `Commit` or `Abort` (Phase 2).
- **The Consumer Trap:** Consumers must never see "Aborted" slots as gaps or garbage. They must skip them efficiently. However, you strictly **cannot** use a simple boolean flag for "validity" because of memory ordering latency.
- **Requirement:** Implement a mechanism (e.g., Dual-Sequence Barriers or Tagged States) that guarantees consumers only process Committed slots in strict order, maintaining Wait-Free progress even if a Producer is stalled mid-transaction.

#### 2. Coroutine-Integrated EBR (The Lifetime Paradox)
Implement **Epoch-Based Reclamation (EBR)** handling for memory safety, but integrated with C++20 Coroutines:
- **The Paradox:** A coroutine holding an Epoch Guard (preventing deletion of objects) might `co_await` (suspend) for network I/O.
- **Constraint:** A suspended coroutine **cannot** hold the global epoch (blocking GC for the whole system). However, if it releases the epoch, the objects it references might be deleted by another thread.
- **Solution:** Implement a custom `awaiter` that automatically exits the epoch on `suspend` and **re-validates** the state (checking for object deletion/version change) upon `resume`. This requires a versioning scheme (e.g., ABA generation counters).

#### 3. AVX-512 Branchless "Shadow Engine" (The SIMD Trap)
Implement a calculation kernel using **Raw AVX-512 Intrinsics** (`<immintrin.h>`):
- The logic involves complex conditional filtering: `if (price > limit && volume < max_vol) { action = buy; } else { action = hold; }`.
- **Constraint:** You are strictly FORBIDDEN from using CPU branching (`if/else`, `switch`, ternary operator) inside the loop.
- **Requirement:** Use **Mask Registers** (`__mmask16`, `_mm512_mask_blend_ps`, etc.) to implement control flow entirely in hardware logic. Code must effectively execute both branches and select the result without pipeline flushes.

#### 4. Compile-Time State Machine
- Use C++23 `concepts` and `consteval` to enforce business logic.
- Transitions between Order States (e.g., `New` -> `Filled`) must be validated at compile-time via type traits. An illegal transition attempt must result in a **compiler error**, not a runtime check.

### ARCHITECTURAL CONSTRAINTS:
1.  **Memory Model:** Use `std::memory_order_acquire` / `release` only. `seq_cst` is banned.
2.  **No Exceptions:** Compile with `-fno-exceptions`. Use `std::expected<T, Error>` for failure handling.
3.  **Strict Aliasing:** Use C++23 `std::start_lifetime_as` for any buffer casting.
4.  **Arena Only:** All coroutine frames must be allocated on a custom monotonic buffer (Arena), not the heap.

### THE "ACID TEST" QUESTION:
In a comment block at the end, answer this specific hardware question based on your code:
*"How does your Transactional Ring Buffer rollback mechanism interact with the L1/L2 Store Buffers? Explain how you prevent a consumer from reading a speculatively written value that was later aborted, specifically in the context of Store-to-Load forwarding failures."*

**Show me the code. Be ruthless with optimization.**
# ROLE: LEAD HFT SYSTEM ARCHITECT & HARDWARE OPTIMIZATION SPECIALIST

**Context:** You are building the core execution unit for "Project Event Horizon" — a high-frequency trading system running on C++23. The goal is not just low latency, but **predictable deterministic latency** under extreme contention.

## TASK: SPECULATIVE ZERO-COPY KERNEL WITH TRANSACTIONAL MEMORY SEMANTICS

Implement a single-file C++23 kernel that handles an event stream.
**The Twist:** The engine operates **speculatively**. It must reserve space and begin processing an order *before* full risk validation is complete. If validation fails, the operation must be **Rolled Back** in a Wait-Free manner without blocking consumers.

### STRICT TECHNICAL REQUIREMENTS:

#### 1. Transactional Lock-Free Ring Buffer (The Logic Trap)
Implement a Disruptor-style Ring Buffer, but with **Atomic Transaction Semantics**:
- **Two-Phase Commit:** A Producer reserves a slot and writes data (Phase 1). It then either `Commit` or `Abort` (Phase 2).
- **The Consumer Trap:** Consumers must never see "Aborted" slots as gaps or garbage. They must skip them efficiently. However, you strictly **cannot** use a simple boolean flag for "validity" because of memory ordering latency.
- **Requirement:** Implement a mechanism (e.g., Dual-Sequence Barriers or Tagged States) that guarantees consumers only process Committed slots in strict order, maintaining Wait-Free progress even if a Producer is stalled mid-transaction.

#### 2. Coroutine-Integrated EBR (The Lifetime Paradox)
Implement **Epoch-Based Reclamation (EBR)** handling for memory safety, but integrated with C++20 Coroutines:
- **The Paradox:** A coroutine holding an Epoch Guard (preventing deletion of objects) might `co_await` (suspend) for network I/O.
- **Constraint:** A suspended coroutine **cannot** hold the global epoch (blocking GC for the whole system). However, if it releases the epoch, the objects it references might be deleted by another thread.
- **Solution:** Implement a custom `awaiter` that automatically exits the epoch on `suspend` and **re-validates** the state (checking for object deletion/version change) upon `resume`. This requires a versioning scheme (e.g., ABA generation counters).

#### 3. AVX-512 Branchless "Shadow Engine" (The SIMD Trap)
Implement a calculation kernel using **Raw AVX-512 Intrinsics** (`<immintrin.h>`):
- The logic involves complex conditional filtering: `if (price > limit && volume < max_vol) { action = buy; } else { action = hold; }`.
- **Constraint:** You are strictly FORBIDDEN from using CPU branching (`if/else`, `switch`, ternary operator) inside the loop.
- **Requirement:** Use **Mask Registers** (`__mmask16`, `_mm512_mask_blend_ps`, etc.) to implement control flow entirely in hardware logic. Code must effectively execute both branches and select the result without pipeline flushes.

#### 4. Compile-Time State Machine
- Use C++23 `concepts` and `consteval` to enforce business logic.
- Transitions between Order States (e.g., `New` -> `Filled`) must be validated at compile-time via type traits. An illegal transition attempt must result in a **compiler error**, not a runtime check.

### ARCHITECTURAL CONSTRAINTS:
1.  **Memory Model:** Use `std::memory_order_acquire` / `release` only. `seq_cst` is banned.
2.  **No Exceptions:** Compile with `-fno-exceptions`. Use `std::expected<T, Error>` for failure handling.
3.  **Strict Aliasing:** Use C++23 `std::start_lifetime_as` for any buffer casting.
4.  **Arena Only:** All coroutine frames must be allocated on a custom monotonic buffer (Arena), not the heap.

### THE "ACID TEST" QUESTION:
In a comment block at the end, answer this specific hardware question based on your code:
*"How does your Transactional Ring Buffer rollback mechanism interact with the L1/L2 Store Buffers? Explain how you prevent a consumer from reading a speculatively written value that was later aborted, specifically in the context of Store-to-Load forwarding failures."*

**Show me the code. Be ruthless with optimization.**
