import torch
import torch.optim as optim

from torch.optim.optimizer import required


class ALIG(optim.Optimizer):
    r"""
    Implements ALIG
    Nesterov momentum is the *standard formula*, and differs
    from pytorch NAG implementation.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        eta (float): initial learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): small constant for numerical stability (default: 1e-5)

    Example:
        >>> optimizer = ALIG(model.parameters(), eta=1, momentum=0.9, weight_decay=1e-4)
        >>> optimizer.zero_grad()
        >>> loss_value = loss_fn(model(input), target)
        >>> loss_value.backward()
        >>> optimizer.step(lambda: float(loss_value))

    .. note::
        In order to compute the step-size, it requires a closure at every step
        that gives the current value of the loss function (without the regularization).
    """

    def __init__(self, params, eta=None, momentum=0, projection_fn=None, eps=1e-5, adjusted_momentum=False):
        if eta is not None and eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(eta=eta, momentum=momentum, step_size=None)
        super(ALIG, self).__init__(params_list, defaults)

        self.adjusted_momentum = adjusted_momentum
        self.projection = projection_fn
        self.eps = eps

        for group in self.param_groups:
            if group['momentum']:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        if self.adjusted_momentum:
            self.apply_momentum = self.apply_momentum_adjusted
        else:
            self.apply_momentum = self.apply_momentum_standard

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def compute_step_size(self, loss):
        # compute squared norm of gradient
        grad_sqrd_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_sqrd_norm += p.grad.data.norm() ** 2

        # compute unclipped step-size
        self.step_size_unclipped = float(loss / (grad_sqrd_norm + self.eps))

        # compute effective step-size (clipped)
        for group in self.param_groups:
            if group["eta"] is not None:
                group["step_size"] = min(self.step_size_unclipped, group["eta"])
            else:
                group["step_size"] = self.step_size_unclipped

        # average step size for monitoring
        self.step_size = sum([g["step_size"] for g in self.param_groups]) / float(len(self.param_groups))

    @torch.autograd.no_grad()
    def step(self, closure):
        loss = closure()

        self.compute_step_size(loss)

        for group in self.param_groups:
            step_size = group["step_size"]
            momentum = group["momentum"]
            for p in group['params']:
                if p.grad is None:
                    continue
                p.add_(-step_size, p.grad)
                # Nesterov momentum
                if momentum:
                    self.apply_momentum(p, step_size, momentum)

        if self.projection is not None:
            self.projection()

    @torch.autograd.no_grad()
    def apply_momentum_standard(self, p, step_size, momentum):
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).add_(-step_size, p.grad)
        p.add_(momentum, buffer)

    @torch.autograd.no_grad()
    def apply_momentum_adjusted(self, p, step_size, momentum):
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).sub_(p.grad)
        p.add_(step_size * momentum, buffer)
