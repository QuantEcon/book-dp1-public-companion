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

This website presents a set of lectures on dynamic programming and its applications in economics, finance, and adjacent fields like operations research, designed and written by [Thomas J. Sargent](http://www.tomsargent.com/) and [John Stachurski](http://johnstachurski.net/). The languages of instruction are Julia and Python.

```{tableofcontents}
```

```{code-cell} julia-1.9
:tags: ["remove-cell"]
using Pkg;
Pkg.activate("./");
Pkg.instantiate();
```