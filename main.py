import numpy as np
import matplotlib.pyplot as plt

from utilities import *


def task_d():
    plt.figure(4)
    plt.title("Numerical and and analytical solutions\nof Lane-Emden where $n=1$ with forwards Euler method")
    for h in [3E-1, 1E-1, 1E-2]:
        xi_list = plot_LE_numerical(n=1, f=LE_diff_auto, h=h, Euler=True)

    plt.plot(xi_list, LE_analytical_n1(xi_list), linestyle="dotted", label="Analytical solution")

    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta$")
    plt.grid()
    plt.legend(loc='lower left')
    plt.xlim([0, None])
    plt.ylim([0, 1.1])

    plt.savefig("Plots\\Task_d.pdf")

    plt.show()


def task_e():
    plt.figure(2)
    plt.title("Numerical solution for Lane-Emden where $n=3/2$")
    plot_LE_numerical(n=3/2, f=LE_diff_auto, h=1E-3, Euler=True)

    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta$")
    plt.grid()
    plt.legend(loc='lower left')
    plt.xlim([0, None])
    plt.ylim([0, 1.1])

    plt.savefig("Plots\\Task_e_1.pdf")

    plt.figure(3)
    plt.title("Numerical solution for Lane-Emden where $n=3$")
    plot_LE_numerical(n=3, f=LE_diff_auto, h=1E-3, Euler=True)

    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta$")
    plt.grid()
    plt.legend(loc='lower left')
    plt.xlim([0, None])
    plt.ylim([0, 1.1])

    plt.savefig("Plots\\Task_e_2.pdf")

    plt.show()


def task_f():
    plt.figure(1)
    plt.title("Numerical and and analytical solutions\nof Lane-Emden where $n=1$ with RK4 method")

    for h in [3E-1, 1E-1, 1E-2]:
        xi_list = plot_LE_numerical(n=1, f=LE_diff_auto, h=h, Euler=False)

    plt.plot(xi_list, LE_analytical_n1(xi_list), linestyle="dotted", label="Analytical solution")

    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta$")
    plt.grid()
    plt.legend(loc='lower left')
    plt.xlim([0, None])
    plt.ylim([0, 1.1])

    plt.savefig("Plots\\Task_f.pdf")

    plt.show()


def task_g():
    global_error_plot()


def task_i(h=None):
    if h is None:
        h = 1E-4  # This value gives RK4 a very small global error, a lot smaller than e.g. h=3E-4
    alpha_list = np.array((0.86, 0.59, 0.0167))

    plt.figure(figsize=(6, 8))

    # --------------------------------------------------------------------------------------------------------
    # Go through alphas and plot two numerical solutions and the one relativistic analytical in each subplot
    # --------------------------------------------------------------------------------------------------------
    for idx, alpha in enumerate(alpha_list):
        plt.subplot(len(alpha_list), 1, idx+1)

        P_bar_init = np.array((P_bar_analytical(alpha, x=0), 0))

        # --------------------------------------------------------------------------------------------------------
        # Plot RK4 result and print global error
        # --------------------------------------------------------------------------------------------------------
        y = numerical_general(h, P_bar_init, P_bar_deriv, alpha,
                              end_condition=lambda y_last: y_last[1] > 1, Euler=False)
        plt.plot(y[:, 1], y[:, 0]/y[0, 0],
                 "--", label=r"RK4".format(alpha, h))

        print(f"Global error for RK4:\t alpha={alpha:0<6}, h={h:.2E}: {y[-1, 0] - P_bar_analytical(alpha, x=1): .1E}")

        # --------------------------------------------------------------------------------------------------------
        # Plot Euler result and print global error
        # --------------------------------------------------------------------------------------------------------
        y = numerical_general(h, P_bar_init, P_bar_deriv, alpha,
                              end_condition=lambda y_last: y_last[1] > 1, Euler=True)
        plt.plot(y[:, 1], y[:, 0] / y[0, 0],
                 "--", label=r"Forw. Euler".format(alpha, h))

        print(f"Global error for Euler:\t alpha={alpha:0<6}, h={h:.2E}: {y[-1, 0] - P_bar_analytical(alpha, x=1): .1E}")

        # --------------------------------------------------------------------------------------------------------
        # Plot analytical solution
        # --------------------------------------------------------------------------------------------------------
        plt.plot(np.linspace(0, 1, 201), P_bar_analytical(alpha, x=np.linspace(0, 1, 201), relative=True),
                 "k", linestyle="dotted", label=r"Analytical".format(alpha), alpha=0.5)

        # Plot config:
        plt.title(r"Relative dimensionless pressure with $\alpha = {}$ and $h=${:.0E}".format(alpha, h))
        plt.ylabel(r"$\bar{P}(x)/\bar{P}(0)$  [1]")
        plt.xlabel(r"$x=r/R$  [1]")
        plt.xlim([0, 1])
        plt.ylim([0, 1.1])
        plt.grid()
        plt.legend()

    plt.tight_layout()

    plt.savefig(f"Plots\\Task_i_h={h:.0E}.pdf")
    plt.show()


def task_j():
    alpha_list = np.array((0.86, 0.59, 0.0167))

    plt.figure(figsize=(6, 8))

    # Go through each alpha value and plot the relativistic and newtonian solution in each subplot
    for idx, alpha in enumerate(alpha_list):
        plt.subplot(len(alpha_list), 1, idx+1)

        # Plot relativistic solution
        plt.plot(np.linspace(0, 1, 201), P_bar_analytical(alpha, x=np.linspace(0, 1, 201), relative=False),  # *4/alpha,
                 "r", label=r"Relativistic".format(alpha), alpha=0.5)

        # Plot newtonian solution
        plt.plot(np.linspace(0, 1, 201), P_bar_newtonian(alpha, x=np.linspace(0, 1, 201), relative=False),  # *4/alpha,
                 "b", label=r"Newtonian".format(alpha), alpha=0.5)

        # Plotting config
        plt.title(r"Dimensionless absolute pressure $\bar{P}$" + r" with $\alpha = {}$".format(alpha))
        plt.ylabel(r"$\bar{P}(x)$  [1]")
        plt.xlabel(r"$x=r/R$  [1]")
        plt.xlim([0, 1])
        plt.ylim([0, None])
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"Plots\\Task_j_unscaled.pdf")
    plt.show()


if __name__ == "__main__":
    # task_d()
    # task_e()
    # task_f()
    # task_g()
    # task_i(h=1E-4)
    # task_j()
    pass
