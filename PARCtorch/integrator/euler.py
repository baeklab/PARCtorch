from PARCtorch.integrator.numintegrator import NumericalIntegrator


class Euler(nint.NumericalIntegrator)
    def __init__(self, **kwarg):
        super(Euler, self).__init__(**kwarg)

    def forward(self, f, current, step_size):
        '''
        Euler integration. Fixed step, 1st order.

        Parameters
        ----------
        f: function, the function that returns time deriviative
        current: tensor, the current state and velocity variables
        step_size: float, integration step size

        Returns
        -------
        final_state: tensor with the same shape of ```current```, the next state and velocity varaibles
        update: tensor with the same shape of ```current```, the update in this step
        '''
        update = f(current)
        final_state = current + step_size*update 
        return final_state, update
