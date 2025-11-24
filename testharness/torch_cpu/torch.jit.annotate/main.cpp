#include "fuzzer_utils.h"
#include <iostream>
#include <string>
#include <torch/jit.h>
#include <torch/script.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        if (Size < 2)
        {
            return 0;
        }

        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        int64_t scalar = 0;
        if (offset < Size)
        {
            scalar = static_cast<int64_t>(Data[offset] % 8);
            ++offset;
        }

        const std::string script = R"JIT(
import torch
from typing import Dict, List, Optional

def annotated_tensor_ops(x: torch.Tensor, val: int):
    # Use torch.jit.annotate to make TorchScript aware of container types.
    base = x.float()
    tensor_list = torch.jit.annotate(List[torch.Tensor], [])
    tensor_list.append(base)
    tensor_dict = torch.jit.annotate(Dict[str, torch.Tensor], {"value": base})
    opt_tensor = torch.jit.annotate(Optional[torch.Tensor], None)
    if opt_tensor is None:
        opt_tensor = base
    annotated_int = torch.jit.annotate(int, val)
    return tensor_list[0] + tensor_dict["value"] + opt_tensor + float(annotated_int)
)JIT";

        auto cu = torch::jit::compile(script);
        auto output = cu->run_method("annotated_tensor_ops", tensor, scalar);
        if (output.isTensor())
        {
            output.toTensor().sum();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
