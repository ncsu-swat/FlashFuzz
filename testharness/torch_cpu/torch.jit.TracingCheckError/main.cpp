#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/tracer.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor if we have more data
        torch::Tensor input2;
        if (offset + 4 < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            input2 = input1.clone();
        }
        
        // Create a simple traced module
        torch::jit::Module traced_module;
        
        try {
            // Define a simple function to trace
            auto add_func = [](torch::Tensor a, torch::Tensor b) {
                return a + b;
            };
            
            // Trace the function
            auto trace = torch::jit::tracer::trace(add_func, {input1, input2});
            traced_module = trace;
            
            // Try to run the traced module with different inputs
            torch::Tensor modified_input1;
            torch::Tensor modified_input2;
            
            // Create modified inputs to trigger tracing check errors
            if (offset + 4 < Size) {
                modified_input1 = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Modify the shape if possible
                if (input1.dim() > 0) {
                    auto sizes = input1.sizes().vec();
                    for (size_t i = 0; i < sizes.size(); i++) {
                        if (sizes[i] > 1) {
                            sizes[i] = sizes[i] - 1;
                            break;
                        } else if (sizes[i] == 0) {
                            sizes[i] = 1;
                            break;
                        }
                    }
                    modified_input1 = torch::zeros(sizes, input1.options());
                } else {
                    modified_input1 = input1 + 1.0;
                }
            }
            
            if (offset + 4 < Size) {
                modified_input2 = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Modify the shape if possible
                if (input2.dim() > 0) {
                    auto sizes = input2.sizes().vec();
                    for (size_t i = 0; i < sizes.size(); i++) {
                        if (sizes[i] > 1) {
                            sizes[i] = sizes[i] - 1;
                            break;
                        } else if (sizes[i] == 0) {
                            sizes[i] = 1;
                            break;
                        }
                    }
                    modified_input2 = torch::zeros(sizes, input2.options());
                } else {
                    modified_input2 = input2 + 1.0;
                }
            }
            
            // Try to run with modified inputs to trigger TracingCheckError
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(modified_input1);
            inputs.push_back(modified_input2);
            
            auto output = traced_module.forward(inputs);
        }
        catch (const std::runtime_error& e) {
            // This catches tracing-related errors
        }
        
        // Try with different control flow paths
        if (offset < Size) {
            uint8_t control_byte = Data[offset++];
            
            try {
                // Define a function with control flow
                auto control_flow_func = [control_byte](torch::Tensor a, torch::Tensor b) {
                    if (control_byte % 2 == 0) {
                        return a + b;
                    } else {
                        return a * b;
                    }
                };
                
                // Trace with one path
                auto trace = torch::jit::tracer::trace(control_flow_func, {input1, input2});
                
                // Run with inputs that might take a different path
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input1);
                inputs.push_back(input2);
                
                auto output = trace.forward(inputs);
            }
            catch (const std::runtime_error& e) {
                // Expected when control flow changes
            }
        }
        
        // Test with empty tensors
        try {
            auto empty_tensor1 = torch::empty({0}, input1.options());
            auto empty_tensor2 = torch::empty({0}, input2.options());
            
            auto empty_func = [](torch::Tensor a, torch::Tensor b) {
                return a + b;
            };
            
            auto trace = torch::jit::tracer::trace(empty_func, {empty_tensor1, empty_tensor2});
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input1);
            inputs.push_back(input2);
            
            auto output = trace.forward(inputs);
        }
        catch (const std::runtime_error& e) {
            // Expected when shapes don't match
        }
        
        // Test with scalar tensors
        try {
            auto scalar_tensor1 = torch::tensor(1.0f, input1.options());
            auto scalar_tensor2 = torch::tensor(2.0f, input2.options());
            
            auto scalar_func = [](torch::Tensor a, torch::Tensor b) {
                return a + b;
            };
            
            auto trace = torch::jit::tracer::trace(scalar_func, {scalar_tensor1, scalar_tensor2});
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input1);
            inputs.push_back(input2);
            
            auto output = trace.forward(inputs);
        }
        catch (const std::runtime_error& e) {
            // Expected when shapes don't match
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
