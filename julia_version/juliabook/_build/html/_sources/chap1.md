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

(getting_started)=
```{raw} html
<div id="qe-notebook-header" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Setting up Your Julia Environment

```{contents} Contents
:depth: 2
```

## Overview

In this lecture we will cover how to get up and running with Julia.

While there are alternative ways to access Julia (e.g. if you have a JupyterHub provided by your university), this section assumes you will install it to your

It is not strictly required for running the lectures, but we will strongly encourage installing and using [Visual Studio Code (VS Code)](https://code.visualstudio.com/).

As the most popular and best-supported open-source code editor, it provides a large number of useful features and extensions.  We will begin to use it as a primary editor in the 

## A Note on Jupyter

Like Python and R, and unlike products such as Matlab and Stata, there is a looser connection between Julia as a programming language and Julia as a specific development environment.

While you will eventually use other editors, there are some advantages to starting with the [Jupyter](http://jupyter.org/) environment while learning Julia.

* The ability to mix formatted text (including mathematical expressions) and code in a single document.
* Nicely formatted output including tables, figures, animation, video, etc.
* Conversion tools to generate PDF slides, static HTML, etc.

We'll discuss the workflow on these features in the