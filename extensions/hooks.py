def gradient_reversal_hook(module, grad_out, grad_in):
    """
    reverses the gradient that is output from the module. can be registered via register_backward_hook for an adversarial classification task
    :param module:
    :param grad_out:
    :param grad_in:
    :return:
    """
   
    return (grad_out[0]*-1,grad_out[1]*-1, grad_out[2]*-1)


class MyHook():
    """
    hook for storing gradients
    """
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()