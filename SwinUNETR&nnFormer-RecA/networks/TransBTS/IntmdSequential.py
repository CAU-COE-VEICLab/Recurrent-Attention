import torch.nn as nn


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input, memory):
        if not self.return_intermediate:
            return super().forward(input, memory)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output, memory = module(output, memory)
            intermediate_outputs[name] = output
        return output, intermediate_outputs

