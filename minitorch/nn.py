from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand

max_reduce = FastOps.reduce(operators.max, -1e9)

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off

def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    
    if not input._tensor.is_contiguous():
            input = input.contiguous()


    #shape into kernel, split height and width
    n_input = input.view(batch, channel, height // kh, kh, width // kw, kw)
    n_input = n_input.permute(0, 1, 2, 4, 3, 5).contiguous()


    tile_input = n_input.view(
        batch, channel, height // kh, width // kw, kh * kw
    )

    return tile_input, int(height // kh), int(width // kw)



def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to the input tensor given kernel size.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, _, _ = input.shape


    result = tile(input, kernel)
    output = result[0].mean(4)


    return output.view(batch, channel, result[1], result[2])




def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to reduce over

    Returns:
    -------
        Tensor of size batch x channel x height x width where argmax is 1, 0 otherwise (one-hot tensor)

    """
    return input == max_reduce(input, dim)

class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for Max is max reduction"""
        result = max_reduce(input, int(dim[0]))
        ctx.save_for_backward(input, result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for Max is argmax"""
        input, old_result = ctx.saved_values
        result = (input == old_result) * grad_output
        return result, 0.0
    

def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction to the input tensor given dimension."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to apply softmax over

    Returns:
    -------
        Tensor of size batch x channel x height x width with softmax applied to dim

    """
    input_exp = input.exp()
    sum_exp_inp = input_exp.sum(dim=dim)
    return input_exp / sum_exp_inp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to apply logsoftmax over

    Returns:
    -------
        Tensor of size batch x channel x height x width with logsoftmax applied to dim
        See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    """
    mx = max(input, dim)
    input_exp = (input - mx).exp()
    sum_exp = input_exp.sum(dim=dim)
    log_sum_exp = sum_exp.log() + mx
    return input - log_sum_exp  


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to the input tensor given kernel size.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, _, _ = input.shape
    middle, new_height, new_width = tile(input, kernel)
    return max(middle, 4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off

    Args:
    ----
        input: batch x channel x height x width
        p: probability of dropout
        ignore: if true, ignore dropout

    Returns:
    -------
        Tensor of size batch x channel x height x width with dropout applied

    """
    if not ignore:
        prob_drop = rand(input.shape)
        level = prob_drop > p

        return input * level
    
    return input