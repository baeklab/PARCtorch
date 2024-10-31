from PARCtorch.integrator.numintegrator import NumericalIntegrator


class Heun(NumericalIntegrator):
    def __init__(self, **kwarg):
        super(Heun, self).__init__(**kwarg)

    def forward(self, f, t, current, step_size):
        """
        Heun integration. Fixed step, explicit, 2nd order.

        Parameters
        ----------
        f: function, the function that returns time deriviative
        t: float, current time
        current: tensor, the current state and velocity variables
        step_size: float, integration step size

        Returns
        -------
        final_state: tensor with the same shape of ```current```, the next state and velocity varaibles
        update: tensor with the same shape of ```current```, the update in this step
        """
        # Compute k1
        k1 = f(t, current)
        # Compute k2
        inp_k2 = current + step_size * k1
        k2 = f(t + step_size, inp_k2)
        # Final
        update = 1 / 2 * (k1 + k2)
        final_state = current + step_size * update
        return final_state, update
