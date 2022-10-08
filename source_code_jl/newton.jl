using PyPlot, LaTeXStrings
using ForwardDiff, Roots
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

x0 = 0.5
g(x) = 1 + (x/(x + 1))
Dg = x -> ForwardDiff.derivative(g, float(x))
g_hat(x; x0=x0) = g(x0) + Dg(x0) * (x - x0)

xs = find_zero(x -> g(x)-x, 0.5)  # find fixed point of g
x1 = (g(x0) - Dg(x0) * x0) / (1 - Dg(x0))

fs = 18

function plot_45(; file_name="../figures/newton_1.pdf",
                 xmin=0.0, xmax=2.6,
                 savefig=false)

    xgrid = LinRange(xmin, xmax, 1000)

    fig, ax = plt.subplots()

    lb_g = L"g"
    ax.plot(xgrid, g.(xgrid),  lw=2, alpha=0.6, label=lb_g)

    lb_g_hat = L"\hat g"
    ax.plot(xgrid, g_hat.(xgrid),  lw=2, alpha=0.6, label=lb_g_hat)

    ax.plot(xgrid, xgrid, "k--", lw=1, alpha=0.7, label=L"45")

    fp1 = (x1,)
    ax.plot(fp1, fp1, "go", ms=5, alpha=0.6)

    ax.plot((x0,), (g_hat(x0),) , "go", ms=5, alpha=0.6)

    ax.plot((xs,), (xs,) , "go", ms=5, alpha=0.6)

    ax.vlines((x0, xs, x1), (0, 0, 0), (g_hat(x0), xs, x1), 
              color="k", linestyle="-.", lw=0.4)



    # ax.annotate(L"k^* = (sA / \delta)^{\frac{1}{1-\alpha}}", 
             # xy=(kstar, kstar),
             # xycoords="data",
             # xytext=(20, -20),
             # textcoords="offset points",
             # fontsize=14)
             # #arrowstyle="->")

    ax.legend(frameon=false, fontsize=fs)

    ax.set_xticks((x0, xs, x1))
    ax.set_xticklabels((L"x_0", L"x^*", L"x_1"), fontsize=fs)
    ax.set_yticks((0, ))

    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, 2.6)
    #ax.set_xlabel(L"x", fontsize=14)

    plt.show()
    if savefig
        fig.savefig(file_name)
    end
end

