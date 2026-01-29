#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to iterate over
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test 1: Iterate over tensor elements using JIT
        // Define a proper TorchScript function (not a method)
        std::string script_code = R"(
def iterate_tensor(x):
    result = []
    for item in x:
        result.append(item)
    return result
)";
        
        try {
            auto compilation_unit = torch::jit::compile(script_code);
            auto &iterate_func = compilation_unit->get_function("iterate_tensor");
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            
            torch::jit::IValue output = iterate_func(inputs);
        } catch (const std::exception&) {
            // Script compilation or execution may fail for certain inputs
        }
        
        // Test 2: Iterate over a list of tensors
        if (offset + 1 < Size) {
            std::vector<torch::Tensor> tensor_list;
            tensor_list.push_back(input_tensor);
            
            // Try to create and add more tensors if we have data
            while (offset + 2 < Size && tensor_list.size() < 5) {
                try {
                    torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    tensor_list.push_back(another_tensor);
                } catch (const std::exception&) {
                    break;
                }
            }
            
            std::string list_script = R"(
def iterate_list(x_list):
    result = []
    for x in x_list:
        result.append(x.sum())
    return result
)";
            
            try {
                auto list_compilation_unit = torch::jit::compile(list_script);
                auto &list_func = list_compilation_unit->get_function("iterate_list");
                
                c10::List<torch::Tensor> jit_list;
                for (const auto& t : tensor_list) {
                    jit_list.push_back(t);
                }
                
                torch::jit::IValue list_input(jit_list);
                torch::jit::IValue list_output = list_func({list_input});
            } catch (const std::exception&) {
                // May fail for incompatible tensors
            }
            
            // Test 3: Nested iteration
            if (tensor_list.size() >= 2) {
                std::string nested_script = R"(
def nested_iterate(x_list):
    result = []
    for x in x_list:
        for item in x:
            result.append(item.sum())
    return result
)";
                
                try {
                    auto nested_compilation_unit = torch::jit::compile(nested_script);
                    auto &nested_func = nested_compilation_unit->get_function("nested_iterate");
                    
                    c10::List<torch::Tensor> nested_list;
                    for (const auto& t : tensor_list) {
                        nested_list.push_back(t);
                    }
                    
                    torch::jit::IValue nested_output = nested_func({torch::jit::IValue(nested_list)});
                } catch (const std::exception&) {
                    // Nested iteration may fail for 0-dim tensors
                }
            }
        }
        
        // Test 4: Enumerate-style iteration with index
        if (offset + 4 < Size) {
            std::string enumerate_script = R"(
def enumerate_iterate(x):
    result = []
    idx = 0
    for item in x:
        result.append(item * idx)
        idx = idx + 1
    return result
)";
            
            try {
                auto enum_compilation_unit = torch::jit::compile(enumerate_script);
                auto &enum_func = enum_compilation_unit->get_function("enumerate_iterate");
                
                torch::jit::IValue enum_output = enum_func({input_tensor});
            } catch (const std::exception&) {
                // May fail for certain tensor types
            }
        }
        
        // Test 5: Range-based iteration in JIT
        if (offset + 2 < Size) {
            int range_end = static_cast<int>(Data[offset++] % 100) + 1;
            
            std::string range_script = R"(
def range_iterate(n: int):
    result = 0
    for i in range(n):
        result = result + i
    return result
)";
            
            try {
                auto range_compilation_unit = torch::jit::compile(range_script);
                auto &range_func = range_compilation_unit->get_function("range_iterate");
                
                torch::jit::IValue range_output = range_func({range_end});
            } catch (const std::exception&) {
                // Range iteration issues
            }
        }
        
        // Test 6: Zip-style iteration (two lists)
        if (offset + 4 < Size) {
            std::string zip_script = R"(
def zip_iterate(list1, list2):
    result = []
    for i in range(min(len(list1), len(list2))):
        result.append(list1[i] + list2[i])
    return result
)";
            
            try {
                auto zip_compilation_unit = torch::jit::compile(zip_script);
                auto &zip_func = zip_compilation_unit->get_function("zip_iterate");
                
                c10::List<torch::Tensor> list1;
                c10::List<torch::Tensor> list2;
                
                list1.push_back(input_tensor);
                
                torch::Tensor t2 = fuzzer_utils::createTensor(Data, Size, offset);
                list2.push_back(t2);
                
                // Add more if possible
                if (offset + 2 < Size) {
                    try {
                        torch::Tensor t3 = fuzzer_utils::createTensor(Data, Size, offset);
                        list1.push_back(t3);
                        if (offset + 2 < Size) {
                            torch::Tensor t4 = fuzzer_utils::createTensor(Data, Size, offset);
                            list2.push_back(t4);
                        }
                    } catch (const std::exception&) {
                    }
                }
                
                torch::jit::IValue zip_output = zip_func({torch::jit::IValue(list1), torch::jit::IValue(list2)});
            } catch (const std::exception&) {
                // Zip iteration may fail
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}