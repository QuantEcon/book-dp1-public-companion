import os
import shutil

root_src_dir = './source_code/'
root_dst_dir = './juliabook/'

shutil.copytree(root_src_dir, root_dst_dir, dirs_exist_ok=True)

#import script_to_myst_Julia as jlscript

def remove_files(root_src_dir, root_dst_dir):
    for file in os.listdir(root_src_dir):
        if os.path.isfile(os.path.join(root_src_dir, file)):
            os.remove(os.path.join(root_dst_dir, file))
        else:
            shutil.rmtree(os.path.join(root_dst_dir, file))

header = """---
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
chapter_meta = {
                "introductions": {
                    "name": "Introductions",
                    "subs": {
                    "two_period_job_search.jl" : ["fig_v1(savefig=true)"],
                    "compute_spec_rad.jl" : [],
                    "power_series.jl" : [],
                    "s_approx.jl" : [],
                    "linear_iter.jl" : [],
                    "linear_iter_fig.jl" : [],
                    "iid_job_search.jl" : ["fig_vseq(savefig=true)", "fig_vstar(savefig=true)"]
                    }
                },
                "operators_fixed_points": {
                    "name": "Operators and Fixed Points",
                    "subs": {
                    "lake.jl" : ["plot_paths(savefig=true)", "plot_growth(savefig=true)"], #to be added to python
                    #"lake_1.jl"
                    }
                },
                "markov_dynamics": {
                    "name": "Markov Dynamics",
                    "subs": {
                    "inventory_sim.jl" : ["model = create_inventory_model()", "plot_ts(model; savefig=true)", "plot_hist(model; savefig=true)"],
                    "is_irreducible.jl" : [],
#                    "inventory_sim.jl",
                    "laborer_sim.jl" : [],
                    "markov_js.jl" : ["plot_main(savefig=true)"],
                    "markov_js_with_sep.jl" : ["plot_main(savefig=true)", "plot_w_stars(savefig=true)"]
                    }
                },
                "optimal_stopping": {
                    "name": "Optimal Stopping",
                    "subs": {
                    "firm_exit.jl" : ["plot_val(savefig=true)", "plot_comparison(savefig=true)"],
                    "american_option.jl" : ["plot_contours(savefig=true)", "plot_strike(savefig=true)"]
                    }
                },
                "markov_decision_processes": {
                    "name": "Markov Decision Processes",
                    "subs": {
                    "inventory_dp.jl" : ["plot_vstar_and_opt_policy(savefig=true)", "plot_ts(savefig=true)"],
                    "finite_opt_saving_0.jl" : [],
                    "finite_opt_saving_1.jl" : [],
                    "finite_opt_saving_2.jl" : ["plot_timing(savefig=true)", "plot_policy()", "plot_time_series(savefig=true)", "plot_histogram(savefig=true)", "plot_lorenz(savefig=true)"],
                    "finite_lq.jl" : ["plot_policy()", "plot_sim(savefig=true)", "plot_timing(savefig=true)"],
                    "firm_hiring.jl" : ["plot_policy()", "plot_sim(savefig=true)", "plot_growth(savefig=true)"],
                    "modified_opt_savings.jl" : ["plot_contours(savefig=true)", "plot_policies(savefig=true)", "plot_time_series(savefig=true)", "plot_histogram(savefig=true)", "plot_lorenz(savefig=true)"]
                    }
                },
                "stochastic_discounting": {
                    "name": "Stochastic Discounting",
                    "subs": {
                    "plot_interest_rates.jl" : ["plot_rates(df_nominal, savefig=true)", "plot_rates(df_real, savefig=true)"], #to be added to python
                    #"plot_interest_rates_real.jl",
                    "pd_ratio.jl" : ["plot_main(savefig=true)"],
                    "inventory_sdd.jl" : ["plot_ts(savefig=true)"]
                    }
                },
                "nonlinear_valuation": {
                    "name": "Nonlinear Valuation",
                    "subs": {
                    "rs_utility.jl" : ["plot_v(savefig=true)", "plot_multiple_v(savefig=true)"],
                    "ez_utility.jl" : ["plot_convergence(savefig=true)", "plot_v(savefig=true)", "vary_gamma(savefig=true)", "vary_alpha(savefig=true)"]
                    }
                },
                "recursive_decision_processes": {
                    "name": "Recursive Decision Processes",
                    "subs": {
                    "quantile_function.jl" : [], #to be added to python
                    "quantile_js.jl" : ["plot_main(savefig=true)"] #to be added to python
                    }
                }
                }

pkg = """```{code-cell} julia-1.9
:tags: [\"remove-cell\"]
using Pkg;
Pkg.activate(\"../\");

using PyCall;
pygui(:tk);
```
"""

#try:
for chapter in chapter_meta.keys():
    chapter_name = chapter_meta[chapter]["name"]
    chapter_subs = chapter_meta[chapter]["subs"]

    with open(f"./juliabook/{chapter}.md", "w", encoding='utf-8') as b:
        b.write(header)
        b.write(f"# {chapter_name}\n\n")
        b.write(contents)
        b.write("\n\n")

        b.write(pkg)

        b.write("\n")

        for sub in chapter_subs.keys():
            with open(f"./juliabook/{sub}", "r", encoding='utf-8') as g:

                b.write(f"#### {sub}\n")
                b.write(f"```{{code-cell}} julia-1.9\n")
                b.write(":tags: [\"hide-input\"]\n")

                text = g.readlines()

                for line in text:
                    if "plt.show()" not in line:
                        b.write(line)
                #b.write(text)

                b.write(f"\n```\n")

                if len(chapter_subs[sub]) > 0:
                    for func_call in chapter_subs[sub]:
                        code = f"\n```{{code-cell}} julia-1.9\n{func_call}\n```\n"
                        

                        b.write(code) 

            
    #return True
#except Exception as e:
    #print("Error in create_myst_nb(): " + str(e))
    #remove_files(root_src_dir, root_dst_dir)
    #print("Failed! - Directory cleaned")
    #return False

#try:
os.system("jupyter-book build juliabook/")
remove_files(root_src_dir, root_dst_dir)
    #print("Success!")
#exit(0)
#except Exception as e:
    #print("Error in create_myst_nb(): " + str(e))
    #remove_files(root_src_dir, root_dst_dir)
    #print("Failed! - Directory cleaned")
    #exit(1)
    
#shutil.copytree("./juliabook/_build/html/", "./docs/", dirs_exist_ok=True)

