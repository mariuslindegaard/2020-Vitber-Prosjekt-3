import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import itertools

XI_INIT = 1E-14


def step_Euler_auto(y, h, f, *args):
    """Performs a single step of forward Euler's method.

    Parameters:
            y: Numerical approximation of y at time t
            h: Step size
            f: RHS of our ODE (RHS = Right hand side). Can be any function that only has y as a variable.
            args: Additional arguments givien to the f-function
        Returns:
            next_y: Numerical approximation of y at time t+h
    """
    return y + h * f(y, *args)


def step_RK4_auto(y, h, f, *args):
    """Performing a single step of the Runge-Kutta fourth order method.

    This is for general autonomous (time-invariant) systems with additional arguments args.
    Parameters:
        f:      RHS of ODEs to be integrated
        y:      numerical approximation of w at time t
        h:      unit/integration step length
        args:   Arguments given to the function f
    Returns:
        numerical approximation of y at time t+h
    """
    s1 = f(y, *args)
    s2 = f(y + (h / 2) * s1, *args)
    s3 = f(y + (h / 2) * s2, *args)
    s4 = f(y + h * s3, *args)
    return y + (h / 6) * (s1 + (2 * s2) + (2 * s3) + s4)


def numerical_general(h, y_init, f, *args, max_iterations=-1, end_condition=lambda y: y!=None, Euler=True):
    """ A full numerical aproximation of an ODE in a set time interval. Performs consecutive Euler steps
    with step size h from start time until the end time. Also takes into account the initial values of the ODE

    Parameters:
            h: Step size
            y_init : Initial condition for y. Typically [theta, chi, xi]
            f: RHS of our ODE
            *args: Additional arguments of function f
            max_iterations : Maximum number of iterations to terminate after
            end_condition : An exit condtition. A function taking in an array of the last calc. values. Same shape as y_init
        Returns:
            y_list: Numerical approximation of all y values. Typically indexed first after iteration, and then [theta, chi, xi]
    """
    try:
        assert max_iterations >= 0 or end_condition(None),\
            "No end condition or max iterations given. Process would not terminate."
    except TypeError:
        pass

    step = step_Euler_auto if Euler else step_RK4_auto

    # Initialise array to store y-values
    y_list = np.zeros((1000, y_init.size))
    # Assign initial condition to theta_0 and chi_0
    y_list[0] = y_init

    last_y = y_init
    idx = 0
    while idx != max_iterations and not end_condition(last_y):
        idx += 1
        # If the array is full, expand it
        if y_list.shape[0] == idx:
            tmp = np.zeros((y_list.shape[0]*2, y_list.shape[1]))
            tmp[:y_list.shape[0]] = y_list
            y_list = tmp
            del tmp

        y_list[idx] = step(y_list[idx-1], h, f, *args)
        last_y = y_list[idx]
    else:
        y_list = y_list[:idx]

    return y_list


def LE_diff_auto(y_array, n):
    """Lane-Emden differential equation as an autonomous system of 1. order diff. equations.

    :param y_array: The arguments as a (3,) np.array wit variables [theta, chi, xi].
    :type y_array: np.ndarray
    :param n: Parameter n in the Lane-Emden equations.
    :type n: float
    :return dy: Derivatives of y_array w.r.t xi.
    :rtype: np.ndarray"""
    theta = max(y_array[0], 0)
    dy = np.asarray([y_array[1],
                     - theta ** n - 2*y_array[1]/y_array[2],
                     1])
    return dy


def LE_analytical_n1(xi_array):
    """Analytical solution of Lane-Emden equation for n=1"""
    xi_array[0] = XI_INIT
    return np.sin(xi_array)/xi_array


def plot_numerical(n, f=LE_diff_auto, h=1E-2, theta_0=1, chi_0=0, start_xi=XI_INIT, Euler=True):

    y = numerical_general(h, np.array((theta_0, chi_0, start_xi)), f, n,
                          end_condition=lambda y_last: y_last[0] <= 0, Euler=Euler)
    theta_list = y[:, 0]
    # chi_list = y[:, 1]
    xi_list = y[:, 2]

    plt.plot(xi_list, theta_list, linestyle='--', label="Numerical solution")

    print("For n =", n, ":  xi_1 =" , xi_list[-1])

    return xi_list


def global_error(h, n, num_type):
    """Find the global error of the function iteration."""
    assert num_type.lower() in ["rk4", "euler"], f"No valid numerical method given: {num_type}"
    assert n in [3/2, 3], f"n is not either 3/2 or 3, but {n}"

    if n == 3/2:
        xi_final = 3.6537537362191657  # 3.65375
    else:
        xi_final = 6.89684861937482  # 6.89685

    y_init = np.array((1, 0, XI_INIT))
    y_array = numerical_general(h, y_init, LE_diff_auto, n,
                                end_condition=lambda y: y[2] > xi_final, Euler=num_type.lower() == "euler")

    # As the range with steps of h generally does not exactly hit xi_final, we weight the last two steps
    # where one of the steps is past xi_final according to how close they both are. (I.e. a linear approximation)
    weight = (y_array[-1, 2] - xi_final) / h
    theta_final = (y_array[-1][0]*(1-weight) + y_array[-2][0]*weight)

    return theta_final - 0  # As theta is 0 with a perfect simulation at xi_final, the error is simply theta


def global_error_plot(show_plot=True, do_parallel=True):
    """Plot the global error of Euler and RK4 as a function of different h-values.

    :param show_plot: Whether to show the plot
    :param do_parallel: Whether to use multiprocessing (approx x2 speedup)
    :return: None
    """

    method_list = ["Euler", "RK4"]
    n_list = np.array([3/2, 3])
    h_list = np.logspace(-4, -1, 17)

    # Create an array of the different parameter combinations available using both methods, n's and all h's.
    # Results for each array is stored in res_array with the same index as in params
    params = list((h, n, method) for method in method_list for n in n_list for h in h_list)
    res_array = np.zeros(len(params))

    # Multiprocessing (approximately) halves the time taken, and it is all  the standard library :D
    if do_parallel:
        with multiprocessing.Pool(8) as Pool:
            results = Pool.starmap(global_error, params)  # Map params to results with global_error function

            for idx, res in enumerate(results):
                res_array[idx] = res  # Put the results in a np.array as they are completed
    else:
        results = itertools.starmap(global_error, params)
        for idx, res in enumerate(results):
            res_array[idx] = res

    # Reshape the res_array to match indexation by parameter list indexes
    res_array = np.abs(
        res_array.reshape((len(method_list), len(n_list), len(h_list)))  # The array is indexed [method][n][h]
                       )

    res_array = np.abs(res_array)

    # Plot the results
    if show_plot:

        plt.figure()
        plt.subplot(211)
        plt.title(r"Error for $n=3/2$")
        plt.xlabel("step length h")
        plt.ylabel("Global error $e_N$")

        plt.loglog(h_list, res_array[0, 0], label="Euler method")
        plt.loglog(h_list, res_array[1, 0], label="RK4 method")
        plt.grid()
        plt.legend()

        plt.subplot(212)
        plt.title(r"Error for $n=3$")
        plt.xlabel("step length h")
        plt.ylabel("Global error $e_N$")

        plt.loglog(h_list, res_array[0, 1], label="Euler method")
        plt.loglog(h_list, res_array[1, 1], label="RK4 method")
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()


def P_bar_deriv(y, alpha):
    """Returns the derivative of the vector [P_bar, x] wrt. x.

    In essence: [dP_bar / dx, 1] from eq. 36
    parameters:
            y: The state vector of the autonomous system [P_bar, x]
            alpha: function parameter (less than 1)

        Returns:
             dy/dx: the derivative of y wrt. x"""

    P_bar = y[0]
    x = y[1]
    return np.array((-(1/2)*alpha*x * (1 + P_bar) * (1 + 3*P_bar) / (1 - alpha*x**2),
                    1))


def P_bar_analytical(alpha, x=np.linspace(0, 1, 101), relative=False):
    """Relativistic analytical solution of TOV equations"""
    b = np.sqrt(1-alpha*x**2)
    c = np.sqrt(1-alpha)
    return (c - b)/(b - 3*c) / (1 if not relative else P_bar_analytical(alpha, 0, relative=False))


def P_bar_newtonian(alpha, x=np.linspace(0, 1, 101), relative=False):
    """Newtonian analytical solution of TOV equations."""
    return alpha * (1-x**2) / 4 / (1 if not relative else alpha/4)


if __name__ == "__main__":
    import timeit

    def test_multi():
        global_error_plot(show_plot=False, do_parallel=True)

    def test_single():
        global_error_plot(show_plot=False, do_parallel=False)

    print("Testing the time taken to do one run of global_error_plot with multiprocessing vs. single process")
    print(f"Timed  multiprocessing: {timeit.timeit(test_multi, number=1): >5.1f} seconds")
    print(f"Timed singleprocessing: {timeit.timeit(test_single, number=1): >5.1f} seconds")