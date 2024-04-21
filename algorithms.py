from scipy.optimize import minimize
import numpy as np

def external_penalty_function_method(
        epsilon,
        c,
        rho,
        x,
        inequality_constraints,
        equality_constraints,
        f,
        max_iter = 100,
        max_opt_iter = 1000,
        callback = None,
):

    # Check algorithm assumptions
    assert c > 1
    assert rho > 0
    assert epsilon > 0

    def penalty(x, current_rho):
        inequality_penalties = np.sum(
            [np.power(np.where(constraint(x) < 0, 0, constraint(x)), 2) for constraint in inequality_constraints],
            axis=0
        )

        equality_penalties = np.sum(
            [np.power(np.abs(constraint(x)), 2) for constraint in equality_constraints],
            axis=0
        )

        return current_rho * (equality_penalties + inequality_penalties)
    
    prev_x = None
    prev_rho = None
    current_step = 1

    while penalty(x, 1) >= epsilon:

        opt_steps = [np.array(x)]
        # Minimize F for current x and rho

        opt_data = minimize(
            method='Powell',
            fun=lambda x: f(x) + penalty(x, rho),
            x0=x,
            options={'maxiter': max_opt_iter},
            callback=opt_steps.append,
        )

        if not opt_data.success:
            print("Opt failure")
            print(opt_data.message)
            break

        if callback:
            callback(
                f,
                penalty,
                rho,
                np.hstack(opt_steps).flatten(),
                current_step,
        )

        rho = c*rho
        x = opt_data.x

        # Between every consequtive steps algorithm should hold those assumptions
        if prev_rho is not None:
            assert penalty(prev_x, 1) >= penalty(x, 1)
            assert f(prev_x) + penalty(prev_x, prev_rho) <= f(x) + penalty(x, rho)
            assert f(prev_x) <= f(x)

        prev_x = x
        prev_rho = rho
        current_step = current_step + 1

        if current_step > max_iter:
            break


    return prev_x


if __name__ == '__main__':
    inequality_constraints_list = [
        lambda x: x-3 # x-3 <= 0
    ]

    equality_constraints_list = [ ]

    INIT_RHO = 1
    INIT_X = 9
    EPS = 10e-8
    C = 2

    solution = external_penalty_function_method(
        EPS,
        C,
        INIT_RHO,
        INIT_X,
        inequality_constraints_list,
        equality_constraints_list,
        f=lambda x: x**2 - 10*x,
    )

    print(solution)
