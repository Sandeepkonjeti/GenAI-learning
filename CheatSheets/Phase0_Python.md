# Phase 0 — Python & CS Fundamentals Cheat Sheet

[← CheatSheets Index](./README.md) | [Full Phase File](../01_Phase0_CS_Fundamentals_and_Python.md)

---

## Key Concepts

| Concept | One-Line Definition |
|---------|-------------------|
| GIL | Global Interpreter Lock — only one thread runs Python bytecode at a time; use `multiprocessing` for CPU tasks |
| Generator | Lazy iterator; `yield` instead of `return`; O(1) memory |
| Decorator | Function wrapper via `@func`; uses closures |
| Context Manager | `__enter__`/`__exit__`; ensures cleanup (files, connections) |
| Coroutine | `async def`; suspends at `await`; requires event loop |
| Big-O | Worst-case time complexity; O(1) < O(log n) < O(n) < O(n log n) < O(n²) |

---

## Python Must-Know Patterns

```python
# --- GENERATORS (memory-safe iteration) ---
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# --- ASYNC I/O (concurrent API calls) ---
import asyncio, aiohttp

async def fetch_all(urls: list[str]) -> list[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(u) for u in urls]
        responses = await asyncio.gather(*tasks)
        return [await r.text() for r in responses]

# --- DATACLASSES ---
from dataclasses import dataclass, field

@dataclass
class Config:
    model: str = "gpt-4o"
    temperature: float = 0.0
    tags: list[str] = field(default_factory=list)

# --- CONTEXT MANAGER ---
from contextlib import contextmanager

@contextmanager
def timer(label: str):
    import time
    t = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - t:.3f}s")
```

---

## NumPy Quick Reference

| Operation | Code | Notes |
|-----------|------|-------|
| Create | `np.zeros((m,n))`, `np.eye(n)`, `np.random.randn(m,n)` | |
| Shape | `a.shape`, `a.reshape(m,n)`, `a.T` | |
| Slice | `a[1:3, ::2]`, `a[mask]` | mask = boolean array |
| Broadcast | `a + b` works if dims align right | (3,4) + (4,) → (3,4) |
| Matmul | `a @ b` or `np.matmul(a,b)` | prefer `@` |
| Einsum | `np.einsum('ij,jk->ik', a, b)` | matmul via einstein notation |
| Axis ops | `np.mean(a, axis=0)` | axis=0 → collapse rows |
| Stack | `np.stack([a,b], axis=0)` | new axis; vs `np.concatenate` |

---

## Complexity Reference

| Structure | Access | Search | Insert | Delete |
|-----------|:------:|:------:|:------:|:------:|
| List | O(1) | O(n) | O(n) | O(n) |
| Dict/Set | O(1)* | O(1)* | O(1)* | O(1)* |
| Sorted list | O(1) | O(log n) | O(n) | O(n) |
| Heap | O(1) min | — | O(log n) | O(log n) |

*amortized

---

## Common Mistakes

- **Mutable default argument**: `def f(x=[])` — use `def f(x=None): x = x or []`
- **`is` vs `==`**: `is` checks identity; `==` checks equality
- **`asyncio.run()` inside Jupyter**: use `await func()` directly in notebook cells
- **Shallow copy**: `b = a.copy()` for lists; nested objects still shared
- **NumPy dtype overflow**: `np.int32` overflows at 2B; use `int64` for large sums
- **Broadcasting error**: shapes must align from the right side

---

## Interview Quick-Hits

**Q: Explain Python's GIL and its implications.**  
A: One thread runs Python bytecode at a time. CPU-bound tasks → `multiprocessing`. I/O-bound → `asyncio` or `threading` (I/O releases GIL).

**Q: When use generator vs list?**  
A: Generator when: data too large for memory, only need to iterate once, producing lazily (pipelines). List when: need random access, multiple iterations, len().

**Q: What is `*args` vs `**kwargs`?**  
A: `*args` = positional varargs tuple; `**kwargs` = keyword varargs dict.

**Q: How does `@lru_cache` work?**  
A: Memoization — caches function return values by argument hash in a dict. Evicts LRU when `maxsize` exceeded.

**Q: What makes NumPy fast?**  
A: Contiguous C array in memory, BLAS/LAPACK for linear algebra, SIMD CPU instructions, avoids Python object overhead.
