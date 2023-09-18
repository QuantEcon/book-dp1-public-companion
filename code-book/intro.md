---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Julia
  language: julia
  name: julia-1.9
---

# Dynamic Programming

**Authors**: [John Stachurski](https://johnstachurski.net/) and [Thomas J. Sargent](http://www.tomsargent.com/)

This website provides Julia and Python code for the textbook [Dynamic Programming](https://dp.quantecon.org/).

```{tableofcontents}
```

```{code-cell} julia-1.9
:tags: ["remove-cell"]
using Pkg;
Pkg.activate("./jl/");
Pkg.instantiate();
```