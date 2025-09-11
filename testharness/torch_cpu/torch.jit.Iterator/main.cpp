#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to iterate over
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a JIT script module to test Iterator
        std::string script_code = R"(
            def forward(self, x):
                result = []
                for item in x:
                    result.append(item)
                return result
        )";
        
        auto compilation_unit = torch::jit::compile(script_code);
        auto module_func = compilation_unit->get_function("forward");
        
        // Create inputs for the module
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // Execute the module with Iterator
        torch::jit::IValue output = module_func(inputs);
        
        // Try different iterator patterns if we have more data
        if (offset + 1 < Size) {
            uint8_t iterator_type = Data[offset++];
            
            // Create a list of tensors to iterate over
            std::vector<torch::Tensor> tensor_list;
            
            // Add the first tensor
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
            
            // Create a JIT script that iterates over a list of tensors
            std::string list_script = R"(
                def forward(self, x_list):
                    result = []
                    for x in x_list:
                        result.append(x.sum())
                    return result
            )";
            
            auto list_compilation_unit = torch::jit::compile(list_script);
            auto list_module_func = list_compilation_unit->get_function("forward");
            
            // Create a list input
            torch::jit::IValue list_input = torch::jit::IValue(tensor_list);
            
            // Execute with the list
            torch::jit::IValue list_output = list_module_func({list_input});
            
            // Test nested iteration if we have enough tensors
            if (tensor_list.size() >= 2) {
                std::string nested_script = R"(
                    def forward(self, x_list):
                        result = []
                        for x in x_list:
                            for item in x:
                                result.append(item)
                        return result
                )";
                
                auto nested_compilation_unit = torch::jit::compile(nested_script);
                auto nested_module_func = nested_compilation_unit->get_function("forward");
                torch::jit::IValue nested_output = nested_module_func({list_input});
            }
        }
        
        // Test with different tensor types if we have more data
        if (offset + 4 < Size) {
            try {
                torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Create a dictionary to iterate over
                c10::Dict<std::string, torch::Tensor> tensor_dict;
                tensor_dict.insert("input", input_tensor);
                tensor_dict.insert("another", another_tensor);
                
                std::string dict_script = R"(
                    def forward(self, x_dict):
                        result = []
                        for key in x_dict:
                            result.append(x_dict[key])
                        return result
                )";
                
                auto dict_compilation_unit = torch::jit::compile(dict_script);
                auto dict_module_func = dict_compilation_unit->get_function("forward");
                torch::jit::IValue dict_input = torch::jit::IValue(tensor_dict);
                torch::jit::IValue dict_output = dict_module_func({dict_input});
            } catch (const std::exception&) {
                // Continue if this part fails
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
