# micrograd — built from scratch

A tiny scalar-valued autograd engine with a small neural-net library on top, implemented
by hand from Andrej Karpathy's micrograd. Implements backpropagation (reverse-mode
autodiff) over a dynamically built DAG, with a PyTorch-like API. The engine is ~90 lines,
the nn library ~65. **Every line typed and understood — not copied.**

Part of Phase 1 of my AI/ML + Hardware Engineering roadmap.

## Example usage

```python
from engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}')   # the result of the forward pass
g.backward()
print(f'{a.grad:.4f}')   # dg/da
print(f'{b.grad:.4f}')   # dg/db
```

## Status

- [x] engine.py — Value class (94 lines) — completed May 2026
- [x] nn.py — Neuron, Layer, MLP classes (60 lines) — completed May 2026
- [ ] Full training demo — in progress
- [ ] Karpathy video walkthrough — in progress

## Reference

- Original: [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
- Video: Andrej Karpathy — *"The spelled-out intro to neural networks and backpropagation: building micrograd"*
