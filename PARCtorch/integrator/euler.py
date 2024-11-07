from PARCtorch.integrator.numintegrator import NumericalIntegrator


class Euler(NumericalIntegrator):
    def __init__(self, **kwarg):
        super(Euler, self).__init__(**kwarg)

    def forward(self, f, t, current, step_size):
        """
        Euler integration. Fixed step, 1st order.

        Args
        f (callable): the function that returns time deriviative
        current (torch.tensor): the current state and velocity variables
        step_size (float): integration step size

        Returns
        final_state (torch.tensor): the same shape as ```current```, the next state and velocity varaibles
        update (torch.tensor): the same shape as ```current```, the updates at this step
        """
        update = f(t, current)
        final_state = current + step_size * update
        return final_state, update
