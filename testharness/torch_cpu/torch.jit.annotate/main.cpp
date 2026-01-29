#include "fuzzer_utils.h"
#include <iostream>
#include <string>
#include <torch/jit.h>
#include <torch/script.h>

// Get compiled functions from a CompilationUnit
static std::shared_ptr<torch::jit::CompilationUnit> getCompiledUnit()
{
    const std::string script = R"JIT(
import torch
from typing import Dict, List, Optional, Tuple

def annotated_tensor_ops(x: torch.Tensor, val: int):
    # Use torch.jit.annotate to make TorchScript aware of container types.
    base = x.float()
    
    # Annotate a List of Tensors
    tensor_list = torch.jit.annotate(List[torch.Tensor], [])
    tensor_list.append(base)
    tensor_list.append(base * 2.0)
    
    # Annotate a Dict of Tensors
    tensor_dict = torch.jit.annotate(Dict[str, torch.Tensor], {})
    tensor_dict["value"] = base
    tensor_dict["doubled"] = base * 2.0
    
    # Annotate Optional Tensor
    opt_tensor = torch.jit.annotate(Optional[torch.Tensor], None)
    if val > 3:
        opt_tensor = base * 3.0
    
    # Annotate primitive types
    annotated_int = torch.jit.annotate(int, val)
    annotated_float = torch.jit.annotate(float, float(val) * 0.5)
    annotated_bool = torch.jit.annotate(bool, val > 2)
    
    # Annotate nested containers
    nested_list = torch.jit.annotate(List[List[int]], [[val, val+1], [val+2]])
    
    # Annotate Tuple
    annotated_tuple = torch.jit.annotate(Tuple[int, float], (val, float(val)))
    
    result = tensor_list[0] + tensor_dict["value"]
    if opt_tensor is not None:
        result = result + opt_tensor
    result = result + annotated_float
    if annotated_bool:
        result = result + 1.0
    
    return result

def annotated_empty_containers(x: torch.Tensor):
    # Test annotating empty containers
    empty_list = torch.jit.annotate(List[torch.Tensor], [])
    empty_dict = torch.jit.annotate(Dict[str, int], {})
    empty_list.append(x)
    empty_dict["key"] = 42
    return empty_list[0] + float(empty_dict["key"])

def annotated_optional_chain(x: torch.Tensor, use_value: bool):
    opt1 = torch.jit.annotate(Optional[torch.Tensor], None)
    opt2 = torch.jit.annotate(Optional[torch.Tensor], x)
    
    if use_value:
        opt1 = x * 2.0
    
    result = torch.zeros_like(x)
    if opt1 is not None:
        result = result + opt1
    if opt2 is not None:
        result = result + opt2
    return result
)JIT";

    return torch::jit::compile(script);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 2)
        {
            return 0;
        }

        size_t offset = 0;
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        int64_t scalar = 0;
        if (offset < Size)
        {
            scalar = static_cast<int64_t>(Data[offset] % 8);
            ++offset;
        }

        uint8_t op_selector = 0;
        if (offset < Size)
        {
            op_selector = Data[offset] % 3;
            ++offset;
        }

        bool use_value = (offset < Size) ? (Data[offset] % 2 == 1) : false;

        // Get the statically compiled unit
        static std::shared_ptr<torch::jit::CompilationUnit> cu = getCompiledUnit();

        torch::IValue output;
        
        switch (op_selector)
        {
        case 0:
        {
            auto func = cu->get_function("annotated_tensor_ops");
            std::vector<torch::IValue> inputs = {tensor, scalar};
            output = func(inputs);
            break;
        }
        case 1:
        {
            auto func = cu->get_function("annotated_empty_containers");
            std::vector<torch::IValue> inputs = {tensor};
            output = func(inputs);
            break;
        }
        case 2:
        {
            auto func = cu->get_function("annotated_optional_chain");
            std::vector<torch::IValue> inputs = {tensor, use_value};
            output = func(inputs);
            break;
        }
        }

        if (output.isTensor())
        {
            // Force computation
            auto result = output.toTensor();
            result.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}