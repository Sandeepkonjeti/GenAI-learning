# Phase 0 Flashcards — Python & CS Fundamentals

[← Flashcards Index](./README.md) | [Phase 0 File](../01_Phase0_CS_Fundamentals_and_Python.md) | [Cheat Sheet](../CheatSheets/Phase0_Python.md)

---

**ID**: P0-001
**Front**: What is Python's Global Interpreter Lock (GIL) and what are its practical implications?
**Back**: The GIL allows only one thread to execute Python bytecode at a time. Implication: threading does NOT provide CPU parallelism in Python. Use `multiprocessing` for CPU-bound tasks. Use `asyncio` or `threading` for I/O-bound tasks (I/O releases the GIL).
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase0 #python #concurrency #GIL

---

**ID**: P0-002
**Front**: What is the difference between `deepcopy` and shallow copy in Python?
**Back**: Shallow copy (`copy.copy()` or `a[:]`): creates new container but references same nested objects. Deep copy (`copy.deepcopy()`): recursively copies all nested objects. Mutable nested objects in shallow copy are shared — modifying them affects both copies.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase0 #python #memory

---

**ID**: P0-003
**Front**: Write a Python generator that yields chunks of size n from a list.
**Back**: `def chunked(lst, n):` `    for i in range(0, len(lst), n):` `        yield lst[i:i+n]` Generators are lazy — O(1) memory regardless of input size. Use `yield` instead of `return`.
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase0 #python #generators

---

**ID**: P0-004
**Front**: What is `asyncio.gather()` and when do you use it?
**Back**: Runs multiple coroutines concurrently and returns their results as a list. Use when you have multiple independent I/O-bound tasks (e.g., 10 API calls). They run concurrently in a single thread via event loop — NOT in parallel. `await asyncio.gather(task1(), task2(), task3())`
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase0 #python #asyncio #concurrency

---

**ID**: P0-005
**Front**: Why is `def f(x=[])` a bug in Python? How do you fix it?
**Back**: Mutable default arguments are created once when the function is defined, not on each call. All calls share the same list object. Fix: `def f(x=None): x = x if x is not None else []`
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase0 #python #gotchas

---

**ID**: P0-006
**Front**: What is `np.einsum('ij,jk->ik', A, B)` computing?
**Back**: Matrix multiplication: $C_{ik} = \sum_j A_{ij} B_{jk}$. Einsum uses Einstein summation notation — repeated index j is summed over. It generalizes many tensor operations (trace: `'ii->'`, outer product: `'i,j->ij'`).
**Difficulty**: Intermediate
**Category**: Math
**Tags**: #phase0 #numpy #einsum #linear-algebra

---

**ID**: P0-007
**Front**: What does NumPy broadcasting do? Give an example.
**Back**: Broadcasting automatically expands dimensions for arithmetic between arrays with compatible shapes. Rule: align shapes from the right; dimensions of size 1 are stretched. Example: `(3,4) + (4,)` → `(4,)` broadcast to `(3,4)` → result is `(3,4)`. Never copies data — virtual expansion.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase0 #numpy #broadcasting

---

**ID**: P0-008
**Front**: What is the time complexity of dict lookup in Python, and why?
**Back**: O(1) average case. Python dicts are hash tables: key → hash → bucket → value. Collision resolution via open addressing. Worst case is O(n) with pathological hash collisions (rare in practice). The hash is computed once per key string.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase0 #python #data-structures #complexity

---

**ID**: P0-009
**Front**: What is `@contextmanager` and how does `yield` work inside it?
**Back**: `@contextmanager` turns a generator function into a context manager. Code before `yield` = `__enter__`. Code after `yield` = `__exit__` (runs even if exception). The yielded value is assigned to `as` target. Used for resource management, timing, transactions.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase0 #python #context-manager

---

**ID**: P0-010
**Front**: Difference between `==` and `is` in Python. When does `is` give surprising results?
**Back**: `==` checks value equality; `is` checks object identity (same memory address). Surprising: `a = 256; b = 256; a is b` → True (CPython caches small integers -5 to 256). `a = 257; b = 257; a is b` → False. Never use `is` for value comparison except `is None`, `is True`, `is False`.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase0 #python #gotchas

---

**ID**: P0-011
**Front**: What is `np.stack` vs `np.concatenate`? When use each?
**Back**: `np.concatenate`: joins arrays along an existing axis (no new dim). `np.stack`: creates a NEW axis and joins along it. Example: two `(3,4)` arrays → `concatenate(axis=0)` → `(6,4)`. `stack(axis=0)` → `(2,3,4)`. Use stack when building batches of same-shaped tensors.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase0 #numpy #shapes

---

**ID**: P0-012
**Front**: What is `@lru_cache` and what are its limitations?
**Back**: Memoization decorator — caches function return values keyed by arguments. Uses a dict (actually OrderedDict) with LRU eviction when `maxsize` is exceeded. Limitations: (1) Arguments must be hashable (no lists/dicts). (2) Cache is per-process (not distributed). (3) No TTL — stale data if underlying data changes.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase0 #python #caching

---

**ID**: P0-013
**Front**: What is the difference between `multiprocessing` and `threading` in Python, and when use each?
**Back**: `threading`: shared memory, GIL means no true CPU parallelism, good for I/O-bound tasks. `multiprocessing`: separate processes with separate memory, true CPU parallelism, good for CPU-bound tasks. `asyncio`: single-threaded concurrency, best for high-concurrency I/O (thousands of connections).
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase0 #python #concurrency

---

**ID**: P0-014
**Front**: What is a decorator in Python and how does it work mechanically?
**Back**: A decorator is a function that takes a function and returns a modified function. `@decorator` is syntactic sugar for `func = decorator(func)`. Uses closures to wrap the original function. `functools.wraps` preserves the original function's `__name__` and docstring in the wrapper.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase0 #python #decorators #closures

---

**ID**: P0-015
**Front**: What is the Big-O complexity of these operations: binary search, quicksort, heapify, dict lookup?
**Back**: Binary search: O(log n). Quicksort: O(n log n) average, O(n²) worst. Heapify: O(n) (not O(n log n) — heapify is O(n)). Heap insert/pop: O(log n). Dict lookup: O(1) average.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase0 #complexity #algorithms

---

**ID**: P0-016
**Front**: How does Python's garbage collector work? What is a reference cycle?
**Back**: CPython uses reference counting as primary GC (immediately collects when refcount → 0). Cyclic GC handles reference cycles (A → B → A, both stuck at refcount=1). `gc` module runs cycle detection. Generational GC: objects that survive collection move to older generations, collected less frequently.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase0 #python #memory-management

---

**ID**: P0-017
**Front**: Write a dataclass with a mutable default field.
**Back**: `from dataclasses import dataclass, field` `@dataclass` `class Config:` `    model: str = "gpt-4o"` `    tags: list[str] = field(default_factory=list)` Use `field(default_factory=list)` — never `tags: list = []` (same mutable default problem as functions).
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase0 #python #dataclasses

---

**ID**: P0-018
**Front**: What is `np.where` and how do you use it?
**Back**: `np.where(condition, x, y)` — element-wise ternary: returns x where condition is True, y elsewhere. Vectorized `if-else`. Example: `np.where(arr > 0, arr, 0)` = ReLU. `np.where(condition)` alone returns indices where True (like `np.argwhere`).
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase0 #numpy

---

**ID**: P0-019
**Front**: Explain Python's `__slots__` and when to use it.
**Back**: `__slots__` replaces the default per-instance `__dict__` with a fixed set of attributes. Benefits: (1) ~40–50% less memory per instance. (2) ~30% faster attribute access. Use when creating millions of instances of the same class. Limitation: cannot add new attributes dynamically.
**Difficulty**: Advanced
**Category**: Theory
**Tags**: #phase0 #python #memory #optimization

---

**ID**: P0-020
**Front**: What is the difference between `iter()` and a generator?
**Back**: Both produce iterators. Generator: defined with `yield`, created by calling a generator function or using generator expression `(x for x in ...)`. `iter()`: creates an iterator from an iterable using `__iter__` protocol. Generators maintain local state between `yield` calls; `iter()` wraps existing iterables.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase0 #python #iterators

---

**ID**: P0-021
**Front**: How do you profile a Python script to find bottlenecks?
**Back**: Line profiler: `@profile` decorator with `kernprof -l script.py`. cProfile: `python -m cProfile -s cumtime script.py`. Memory: `memory_profiler`. For NumPy: `%timeit` in Jupyter. Production: `py-spy` (statistical profiler, no code changes needed). Rule: profile before optimizing — don't guess.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase0 #python #profiling #optimization

---

**ID**: P0-022
**Front**: What is `*args` and `**kwargs`? Give a use case.
**Back**: `*args` collects extra positional arguments into a tuple. `**kwargs` collects extra keyword arguments into a dict. Use case: decorator that passes arguments through: `def decorator(func):` `    def wrapper(*args, **kwargs):` `        return func(*args, **kwargs)`. Also: partial function application, API wrappers.
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase0 #python

---

**ID**: P0-023
**Front**: What does `np.newaxis` do? Give an example.
**Back**: `np.newaxis` is an alias for `None`. When used in indexing, it inserts a new dimension of size 1. Example: `v.shape == (4,)` → `v[np.newaxis, :].shape == (1, 4)` → `v[:, np.newaxis].shape == (4, 1)`. Useful for enabling broadcasting between 1D vectors.
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase0 #numpy #broadcasting

---

**ID**: P0-024
**Front**: What is Git rebase vs merge? When would you use each?
**Back**: Merge: creates a merge commit preserving full branch history. Rebase: moves commits to new base, creating linear history. Use rebase to clean up local feature branch before PR (linear, readable history). Use merge to preserve true history of feature development. Never rebase shared/public branches.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase0 #git

---

**ID**: P0-025
**Front**: How do you handle API secrets securely in Python?
**Back**: (1) Store in environment variables. (2) Load with `os.environ["OPENAI_API_KEY"]` or `python-dotenv`. (3) `.env` file in `.gitignore`. Never: hardcode in source, commit to Git, log in plaintext. Production: use secrets manager (AWS Secrets Manager, Azure Key Vault, Databricks secrets).
**Difficulty**: Beginner
**Category**: Production
**Tags**: #phase0 #python #security

---

**ID**: P0-026
**Front**: What is `np.broadcast_to` and when is it more efficient than `np.tile`?
**Back**: `np.broadcast_to(arr, shape)` creates a read-only view with broadcasted shape — zero memory copy. `np.tile` copies data. For read-only operations, `broadcast_to` is O(1) memory. Example: broadcast a (d,) vector to (B, T, d) without copying: `np.broadcast_to(vec, (B, T, d))`.
**Difficulty**: Advanced
**Category**: Coding
**Tags**: #phase0 #numpy #memory

---

**ID**: P0-027
**Front**: What is a Python `__enter__`/`__exit__` protocol used for?
**Back**: The context manager protocol used with `with` statements. `__enter__` runs at the start of the block, returns the bound value. `__exit__(exc_type, exc_val, exc_tb)` runs at end — handles cleanup even if exception occurs. Return `True` from `__exit__` to suppress the exception.
**Difficulty**: Intermediate
**Category**: Theory
**Tags**: #phase0 #python #context-manager

---

**ID**: P0-028
**Front**: When should you use `np.float32` vs `np.float64` in ML?
**Back**: `float32` (4 bytes): use for ML/DL — sufficient precision, 2× faster on GPUs, 2× less memory. `float64` (8 bytes): use for scientific computing requiring high precision, or when accumulating many small errors. PyTorch defaults to float32. NumPy defaults to float64. Always explicit: `arr.astype(np.float32)`.
**Difficulty**: Intermediate
**Category**: Production
**Tags**: #phase0 #numpy #precision

---

**ID**: P0-029
**Front**: What is `functools.partial` and give a use case?
**Back**: Creates a new function with some arguments pre-filled. Example: `from functools import partial` `add = lambda x, y: x + y` `add5 = partial(add, 5)` `add5(3)  # 8`. Use case: create specialized functions from general ones for callbacks, configurers.
**Difficulty**: Beginner
**Category**: Coding
**Tags**: #phase0 #python #functional

---

**ID**: P0-030
**Front**: What is the difference between `np.dot`, `np.matmul`, and `@`?
**Back**: For 2D arrays: all three are equivalent (matrix multiply). For 1D arrays: `np.dot` computes inner product (scalar); `@` and `matmul` also compute inner product. For N-D arrays: `matmul`/`@` broadcast over batch dims; `np.dot` sums over last axis of first array and second-to-last of second. Prefer `@` for clarity.
**Difficulty**: Intermediate
**Category**: Coding
**Tags**: #phase0 #numpy #linear-algebra
