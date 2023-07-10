header = """
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
"""

contents = """
```{contents} Contents
:depth: 2
```
"""
chapter_meta = {"introductions" : {"name" : "Introductions", "subs" : ["two_period_job_search.jl", "compute_spec_rad.jl", "power_series.jl", "s_approx.jl", "linear_iter.jl", "iid_job_search.jl"]},
"operators_fixed_points" : {"name" : "Operators and Fixed Points", "subs" : ["lake_2.jl", "lake_1.jl"]},
"markov_dynamics": {"name" : "Markov Dynamics", "subs" : ["inventory_sim.jl", "is_irreducible.jl", "inventory_sim.jl", "laborer_sim.jl", "markov_js.jl", "markov_js_with_sep.jl"]},
"optimal_stopping" : {"name" : "Optimal Stopping", "subs" : ["firm_exit.jl", "american_option.jl"]},
"markov_decision_processes" : {"name" : "Markov Decision Processes", "subs" : ["inventory_dp.jl", "finite_opt_saving_0.jl", "finite_opt_saving_1.jl", "finite_opt_saving_2.jl", "finite_lq.jl", "firm_hiring.jl", "modified_opt_savings.jl"]},
"stochastic_discounting" : {"name" : "Stochastic Discounting", "subs" : ["plot_interest_rates_nominal.jl", "plot_interest_rates_real.jl", "pd_ratio.jl", "inventory_sdd.jl"]},
"nonlinear_valuation" : {"name" : "Nonlinear Valuation", "subs" : ["rs_utility.jl", "ez_utility.jl", ]}, 
"recursive_decision_processes" : {"name" : "Recursive Decision Processes", "subs" : ["quantile_function.jl", "quantile_js.jl"]}}

def create_myst_nb():
  try:
    for chapter in chapter_meta.keys():
        chapter_name = chapter_meta[chapter]["name"]
        chapter_subs = chapter_meta[chapter]["subs"]
        
        with open(f"{chapter}.md", "w") as f:
            f.write(header)
            f.write(f"# {chapter_name}\n\n")
            f.write(contents)
            f.write("\n\n")
            for sub in chapter_subs:
                f.write(f"#### {sub}\n")
                f.write(f"```{{code-cell}} julia-1.9\n")
                f.write(":tags: [\"hide-input\"]\n\n")

                with open(sub, "r") as g:
                    f.write(g.read())
                f.write(f"```\n\n")
  except Exception as e:
    print("Error in create_myst_nb(): " + str(e))
