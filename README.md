# Micrograd — Built from Scratch

Andrej Karpathy's micrograd implemented manually as part of my 
AI/ML engineering roadmap. Every line typed and understood — 
not copied.

## What is micrograd?
A tiny autograd engine that implements backpropagation over a 
dynamically built DAG. Trains neural networks using only Python 
and numpy — no PyTorch, no frameworks.

## Status
- [x] engine.py — Value class (94 lines) — completed May 2026
- [x] nn.py — Neuron, Layer, MLP classes (60 lines) — completed May 2026
- [ ] Full training demo — in progress
- [ ] Karpathy video walkthrough — in progress

## What I learned
- How backpropagation works at the mathematical level
- How gradients flow through addition, multiplication, power, and ReLU
- How neurons, layers, and MLPs are built from scratch
- The forward pass and backward pass in full detail

## Reference
- Original: github.com/karpathy/micrograd
- Video: Andrej Karpathy — "The spelled-out intro to neural networks"
- Roadmap: AI/ML + Hardware Engineering — Phase 1
