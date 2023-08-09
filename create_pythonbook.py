import os
import shutil

root_src_dir = './source_code_py/'
root_dst_dir = './python_version/pythonbook/'

shutil.copytree(root_src_dir, root_dst_dir, dirs_exist_ok=True)


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
  display_name: Python 3
  language: python
  name: python3
---

(placeholder_text)=
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
        "name": "Chapter 1: Introductions",
        "subs": { # order determines order in chapter
            "two_period_job_search.py": ["fig_v1(savefig=True)"],
            "compute_spec_rad.py": [],
            "power_series.py": [],
            "s_approx.py": [],
            "linear_iter.py": [],
            "linear_iter_fig.py": [],
            "iid_job_search.py": ["fig_vseq(savefig=True)", "fig_vstar(savefig=True)"]  # plotting functions are added here to keep source code clean
        }
    },
    "operators_fixed_points": {
        "name": "Chapter 2: Operators and Fixed Points",
        "subs": {
            "lake.py": ["plot_paths(savefig=True)", "plot_growth(savefig=True)"],
            # "lake_1.py"
        }
    },
    "markov_dynamics": {
        "name": "Chapter 3: Markov Dynamics",
        "subs": {
            "inventory_sim.py": ["model = create_inventory_model()", "plot_ts(model, savefig=True)", "plot_hist(model, savefig=True)"],
            "is_irreducible.py": [],
            #                    "inventory_sim.py",
            "laborer_sim.py": [],
            "markov_js.py": ["plot_main(savefig=True)"],
            "markov_js_with_sep.py": ["plot_main(savefig=True)", "plot_w_stars(savefig=True)"]
        }
    },
    "optimal_stopping": {
        "name": "Chapter 4: Optimal Stopping",
        "subs": {
            "firm_exit.py": ["plot_val(savefig=True)", "plot_comparison(savefig=True)"],
            "american_option.py": ["plot_contours(savefig=True)", "plot_strike(savefig=True)"]
        }
    },
    "markov_decision_processes": {
        "name": "Chapter 5: Markov Decision Processes",
        "subs": {
            "inventory_dp.py": ["plot_vstar_and_opt_policy(savefig=True)", "plot_ts(savefig=True)"],
            "finite_opt_saving_0.py": [],
            "finite_opt_saving_1.py": [],
            "finite_opt_saving_2.py": ["plot_timing(savefig=True)", "plot_policy()", "plot_time_series(savefig=True)", "plot_histogram(savefig=True)", "plot_lorenz(savefig=True)"],
            "finite_lq.py": ["plot_policy()", "plot_sim(savefig=True)", "plot_timing(savefig=True)"],
            "firm_hiring.py": ["plot_policy()", "plot_sim(savefig=True)", "plot_growth(savefig=True)"],
            "modified_opt_savings.py": ["plot_contours(savefig=True)", "plot_policies(savefig=True)", "plot_time_series(savefig=True)", "plot_histogram(savefig=True)", "plot_lorenz(savefig=True)"]
        }
    },
    "stochastic_discounting": {
        "name": "Chapter 6: Stochastic Discounting",
        "subs": {
            "plot_interest_rates.py": ["plot_rates(df_nominal, savefig=True)", "plot_rates(df_real, savefig=True)"],
            # "plot_interest_rates_real.py",
            "pd_ratio.py": ["plot_main(savefig=True)"],
            "inventory_sdd.py": ["plot_ts(savefig=True)"]
        }
    },
    "nonlinear_valuation": {
        "name": "Chapter 7: Nonlinear Valuation",
        "subs": {
            "rs_utility.py": ["plot_v(savefig=True)", "plot_multiple_v(savefig=True)"],
            "ez_utility.py": ["plot_convergence(savefig=True)", "plot_v(savefig=True)", "vary_gamma(savefig=True)", "vary_alpha(savefig=True)"]
        }
    },
    "recursive_decision_processes": {
        "name": "Chapter 8: Recursive Decision Processes",
        "subs": {
            "quantile_function.py": [],
            "quantile_js.py": ["plot_main(savefig=True)"]
        }
    }
}

try:
    for chapter in chapter_meta.keys():
        chapter_name = chapter_meta[chapter]["name"]
        chapter_subs = chapter_meta[chapter]["subs"]

        with open(f"./python_version/pythonbook/{chapter}.md", "w", encoding='utf-8') as b:
            new_header = header.replace("placeholder_text", chapter_name)
            b.write(new_header)
            b.write(f"# {chapter_name}\n\n")
            b.write(contents)
            b.write("\n\n")

            b.write("\n")

            for sub in chapter_subs.keys():
                with open(f"./python_version/pythonbook/{sub}", "r", encoding='utf-8') as g:

                    b.write(f"#### {sub}\n")
                    b.write(f"```{{code-cell}} python\n")
                    b.write(":tags: [\"hide-input\"]\n")

                    text = g.readlines()

                    for line in text:
                        if "plt.show()" not in line:
                            b.write(line)

                        if "import matplotlib.pyplot as plt" in line:
                            b.write(line + "\nplt.rcParams.update({\"text.usetex\": True, \"font.size\": 14})\n")

                    b.write(f"\n```\n")

                    if len(chapter_subs[sub]) > 0:
                        for func_call in chapter_subs[sub]:
                            code = f"\n```{{code-cell}} python\n{func_call}\n```\n"

                            b.write(code)
except Exception as e:
    print("Error in markdown generation: " + str(e))
    remove_files(root_src_dir, root_dst_dir)
    print("Failed! - Directory cleaned")
    exit(1)

try:
    os.system("jupyter-book build ./python_version/pythonbook/")
    remove_files(root_src_dir, root_dst_dir)
    print("Success!")
    exit(0)
except Exception as e:
    print("Error in building Jupyter Book: " + str(e))
    remove_files(root_src_dir, root_dst_dir)
    print("Failed! - Directory cleaned")
    exit(1)
