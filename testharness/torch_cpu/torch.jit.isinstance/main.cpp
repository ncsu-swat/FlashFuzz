#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // torch.jit.isinstance is a TorchScript-only construct for runtime type checking
        // It can only be used inside TorchScript code, not as a direct C++ API call
        
        // Test 1: Basic isinstance checks with various types
        static std::string script_code = R"JIT(
import torch
from typing import Dict, List, Optional, Tuple

def check_tensor(x: torch.Tensor) -> bool:
    return isinstance(x, torch.Tensor)

def check_optional_tensor(x: Optional[torch.Tensor]) -> bool:
    return isinstance(x, torch.Tensor)

def check_int_value(x: int) -> bool:
    return isinstance(x, int)

def check_float_value(x: float) -> bool:
    return isinstance(x, float)

def check_bool_value(x: bool) -> bool:
    return isinstance(x, bool)

def check_str_value(x: str) -> bool:
    return isinstance(x, str)

def check_list_int(x: List[int]) -> bool:
    return isinstance(x, List[int])

def check_list_tensor(x: List[torch.Tensor]) -> bool:
    return isinstance(x, List[torch.Tensor])

def check_dict_str_int(x: Dict[str, int]) -> bool:
    return isinstance(x, Dict[str, int])

def check_tuple_int_int(x: Tuple[int, int]) -> bool:
    return isinstance(x, Tuple[int, int])
)JIT";

        static std::shared_ptr<torch::jit::CompilationUnit> cu = torch::jit::compile(script_code);

        // Create tensor from fuzzer input
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Test isinstance with Tensor
        {
            std::vector<torch::jit::IValue> inputs = {tensor};
            auto& func = cu->get_function("check_tensor");
            torch::jit::IValue result = func(inputs);
            (void)result.toBool();
        }

        // Test isinstance with int
        if (offset < Size) {
            int64_t int_val = static_cast<int64_t>(Data[offset++] % 128);
            std::vector<torch::jit::IValue> inputs = {int_val};
            auto& func = cu->get_function("check_int_value");
            torch::jit::IValue result = func(inputs);
            (void)result.toBool();
        }

        // Test isinstance with float
        if (offset + sizeof(float) <= Size) {
            float float_val;
            std::memcpy(&float_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize float value
            if (std::isnan(float_val) || std::isinf(float_val)) {
                float_val = 0.0f;
            }
            std::vector<torch::jit::IValue> inputs = {static_cast<double>(float_val)};
            auto& func = cu->get_function("check_float_value");
            torch::jit::IValue result = func(inputs);
            (void)result.toBool();
        }

        // Test isinstance with bool
        if (offset < Size) {
            bool bool_val = (Data[offset++] % 2) == 1;
            std::vector<torch::jit::IValue> inputs = {bool_val};
            auto& func = cu->get_function("check_bool_value");
            torch::jit::IValue result = func(inputs);
            (void)result.toBool();
        }

        // Test isinstance with string
        if (offset < Size) {
            size_t str_len = std::min(static_cast<size_t>(Data[offset++] % 16), Size - offset);
            std::string str_val(reinterpret_cast<const char*>(Data + offset), str_len);
            offset += str_len;
            std::vector<torch::jit::IValue> inputs = {str_val};
            auto& func = cu->get_function("check_str_value");
            torch::jit::IValue result = func(inputs);
            (void)result.toBool();
        }

        // Test isinstance with List[int]
        if (offset < Size) {
            c10::List<int64_t> int_list;
            size_t list_size = Data[offset++] % 5;
            for (size_t i = 0; i < list_size && offset < Size; i++) {
                int_list.push_back(static_cast<int64_t>(Data[offset++]));
            }
            std::vector<torch::jit::IValue> inputs = {int_list};
            auto& func = cu->get_function("check_list_int");
            torch::jit::IValue result = func(inputs);
            (void)result.toBool();
        }

        // Test isinstance with List[Tensor]
        if (offset < Size) {
            c10::List<torch::Tensor> tensor_list;
            size_t list_size = Data[offset++] % 3 + 1;
            for (size_t i = 0; i < list_size && offset + 4 < Size; i++) {
                torch::Tensor small_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensor_list.push_back(small_tensor);
            }
            std::vector<torch::jit::IValue> inputs = {tensor_list};
            auto& func = cu->get_function("check_list_tensor");
            torch::jit::IValue result = func(inputs);
            (void)result.toBool();
        }

        // Test isinstance with Dict[str, int]
        if (offset + 2 < Size) {
            c10::Dict<std::string, int64_t> dict;
            size_t dict_size = Data[offset++] % 3;
            for (size_t i = 0; i < dict_size && offset + 1 < Size; i++) {
                std::string key = "key" + std::to_string(Data[offset++] % 10);
                int64_t value = static_cast<int64_t>(Data[offset++]);
                dict.insert(key, value);
            }
            std::vector<torch::jit::IValue> inputs = {dict};
            auto& func = cu->get_function("check_dict_str_int");
            torch::jit::IValue result = func(inputs);
            (void)result.toBool();
        }

        // Test isinstance with Tuple[int, int]
        if (offset + 2 <= Size) {
            int64_t a = static_cast<int64_t>(Data[offset++]);
            int64_t b = static_cast<int64_t>(Data[offset++]);
            auto tuple_val = c10::ivalue::Tuple::create({a, b});
            std::vector<torch::jit::IValue> inputs = {tuple_val};
            auto& func = cu->get_function("check_tuple_int_int");
            torch::jit::IValue result = func(inputs);
            (void)result.toBool();
        }

        // Test 2: isinstance with union types (using Optional as example)
        if (offset < Size) {
            bool use_none = (Data[offset++] % 2) == 0;
            torch::jit::IValue optional_tensor;
            if (use_none) {
                optional_tensor = torch::jit::IValue();
            } else {
                optional_tensor = tensor;
            }
            std::vector<torch::jit::IValue> inputs = {optional_tensor};
            auto& func = cu->get_function("check_optional_tensor");
            try {
                torch::jit::IValue result = func(inputs);
                (void)result.toBool();
            } catch (...) {
                // Optional with None may fail type check - expected
            }
        }

        // Test 3: More complex isinstance usage in control flow
        static std::string complex_script = R"JIT(
import torch
from typing import List

def process_by_type(x: List[int], y: torch.Tensor) -> torch.Tensor:
    if isinstance(x, List[int]):
        scale = float(len(x))
    else:
        scale = 1.0
    if isinstance(y, torch.Tensor):
        return y * scale
    return y
)JIT";

        static std::shared_ptr<torch::jit::CompilationUnit> cu2 = torch::jit::compile(complex_script);

        if (offset < Size) {
            c10::List<int64_t> int_list;
            size_t list_size = Data[offset++] % 5 + 1;
            for (size_t i = 0; i < list_size && offset < Size; i++) {
                int_list.push_back(static_cast<int64_t>(Data[offset++]));
            }
            std::vector<torch::jit::IValue> inputs = {int_list, tensor};
            auto& func = cu2->get_function("process_by_type");
            torch::jit::IValue result = func(inputs);
            (void)result.toTensor();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}