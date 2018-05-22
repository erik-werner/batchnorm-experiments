import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class BatchNorm2d(torch.nn.Module):
    """torch.nn.BatchNorm2d, with added parameter use_batch_averages
    The wrapper code is copied from the standard implementation.

    If use_batch_averages is True, this layer should be identical to
    torch.nn.BatchNorm2d, except for rounding errors. Also it's much slower.

    If use_batch_averages is False, then use the running averages for inference,
    and fake the backward pass by assuming that the running averages and batch
    averages are identical.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, use_batch_averages=True):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.use_batch_averages = use_batch_averages
        assert track_running_stats or use_batch_averages
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.training and not self.use_batch_averages:
            return self._forward_training_running(input)
        elif self.training or not self.track_running_stats:
            return self._forward_training_batch(input)
        else:
            return self._forward_eval(input)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine}, track_running_stats={track_running_stats})'
                .format(name=self.__class__.__name__, **self.__dict__))

    def _forward_training_batch(self, input):
        # There's probably a simpler way to do the reshaping.
        input_reshaped = input.transpose(0, 1).contiguous().view(input.size(1), -1)
        batch_mean = input_reshaped.mean(1)
        batch_var = input_reshaped.var(1, unbiased=False)

        if self.track_running_stats:
            self.running_mean += self.momentum * (batch_mean.data - self.running_mean)
            self.running_var += self.momentum * (batch_var.data - self.running_var)

        return self._get_output(input, batch_mean, batch_var)

    def _forward_training_running(self, input):
        input_reshaped = input.transpose(0, 1).contiguous().view(input.size(1), -1)
        batch_mean = input_reshaped.mean(1)
        batch_var = input_reshaped.var(1, unbiased=False)

        self.running_mean += self.momentum * (batch_mean.data - self.running_mean)
        self.running_var += self.momentum * (batch_var.data - self.running_var)

        mean = self.running_mean[:, None, None]
        std = Variable((self.running_var[:, None, None] + self.eps).sqrt())
        normalize_input = ShiftAndScale.apply
        normalized_input = normalize_input(input, mean, std)

        return self.weight[:, None, None] * normalized_input + self.bias[:, None, None]

    def _forward_eval(self, input):
        return self._get_output(input, Variable(self.running_mean),
                                Variable(self.running_var))

    def _get_output(self, input, mean, var):
        scaled_input = (input - mean[:, None, None]) \
                        / (var + self.eps).sqrt()[:, None, None]
        return self.weight[:, None, None] * scaled_input + self.bias[:, None, None]


class ShiftAndScale(torch.autograd.Function):
    """
    In the forward pass, shift and scale according to the parameters mean and std.

    In the backward pass, we assume that mean and std correspond to the batch mean
    and batch std, and return the gradients of the shift and scale operation under
    this assumption.
    """

    @staticmethod
    def forward(ctx, input, mean, std):
        normalized_input = (input - mean) / std
        ctx.save_for_backward(normalized_input, std)
        return normalized_input

    @staticmethod
    def backward(ctx, grad_output):
        normalized_input, std = ctx.saved_variables
        grad_output_mean = (grad_output.transpose(0, 1).
                            contiguous().view(grad_output.size(1), -1).
                            mean()[:, None, None])
        var_mean = ((normalized_input * grad_output).transpose(0, 1).
                    contiguous().view(grad_output.size(1), -1).mean(1)[:, None, None])
        grad_input = (grad_output - grad_output_mean - normalized_input * var_mean) / std
        return grad_input, None, None
