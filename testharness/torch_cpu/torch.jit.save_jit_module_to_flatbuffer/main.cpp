#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <fstream>        // For file operations
#include <string>         // For string operations

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
        
        // Create a simple module with a tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple JIT module
        struct TestModule : torch::nn::Module {
            TestModule(torch::Tensor tensor) {
                register_parameter("weight", tensor);
            }
            
            torch::Tensor forward(torch::Tensor input) {
                return input + named_parameters()["weight"];
            }
        };
        
        // Create the module with our tensor
        auto module = std::make_shared<TestModule>(input_tensor);
        
        // Script the module
        torch::jit::Module script_module = torch::jit::trace(module, input_tensor);
        
        // Create a temporary file path for saving
        std::string temp_file = "temp_module.pt";
        
        // Save the module to a file
        script_module.save(temp_file);
        
        // Try to load the module back to verify it worked
        if (offset < Size) {
            // Use remaining data to determine if we should try loading
            uint8_t should_load = Data[offset++];
            if (should_load % 2 == 0) {
                // Try loading the module back
                auto loaded_module = torch::jit::load(temp_file);
                
                // Try running the loaded module with a test input
                torch::Tensor test_input;
                if (offset < Size) {
                    test_input = fuzzer_utils::createTensor(Data, Size, offset);
                } else {
                    test_input = torch::ones_like(input_tensor);
                }
                
                // Run the loaded module
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(test_input);
                auto output = loaded_module.forward(inputs);
            }
        }
        
        // Try to remove the temporary file
        std::remove(temp_file.c_str());
        
        // Try additional edge cases if we have more data
        if (offset + 2 < Size) {
            uint8_t edge_case = Data[offset++];
            
            // Try saving with different options based on remaining data
            switch (edge_case % 4) {
                case 0: {
                    // Try saving an empty module
                    struct EmptyModule : torch::nn::Module {
                        torch::Tensor forward(torch::Tensor input) {
                            return input;
                        }
                    };
                    auto empty_module = std::make_shared<EmptyModule>();
                    torch::jit::Module empty_script = torch::jit::trace(empty_module, input_tensor);
                    empty_script.save("empty_module.pt");
                    std::remove("empty_module.pt");
                    break;
                }
                case 1: {
                    // Try saving to a non-existent directory
                    try {
                        script_module.save("/nonexistent/dir/module.pt");
                    } catch (...) {
                        // Expected to fail, ignore
                    }
                    break;
                }
                case 2: {
                    // Try saving a module with multiple parameters
                    struct MultiParamModule : torch::nn::Module {
                        MultiParamModule(torch::Tensor t1, torch::Tensor t2) {
                            register_parameter("weight1", t1);
                            register_parameter("weight2", t2);
                        }
                        
                        torch::Tensor forward(torch::Tensor input) {
                            return input + named_parameters()["weight1"] + named_parameters()["weight2"];
                        }
                    };
                    
                    torch::Tensor t2;
                    if (offset < Size) {
                        t2 = fuzzer_utils::createTensor(Data, Size, offset);
                    } else {
                        t2 = torch::ones_like(input_tensor);
                    }
                    
                    auto multi_module = std::make_shared<MultiParamModule>(input_tensor, t2);
                    torch::jit::Module multi_script = torch::jit::trace(multi_module, input_tensor);
                    multi_script.save("multi_module.pt");
                    std::remove("multi_module.pt");
                    break;
                }
                case 3: {
                    // Try saving a module with a method that takes multiple inputs
                    struct MultiInputModule : torch::nn::Module {
                        MultiInputModule(torch::Tensor tensor) {
                            register_parameter("weight", tensor);
                        }
                        
                        torch::Tensor forward(torch::Tensor input1, torch::Tensor input2) {
                            return input1 + input2 + named_parameters()["weight"];
                        }
                    };
                    
                    auto multi_input_module = std::make_shared<MultiInputModule>(input_tensor);
                    std::vector<torch::Tensor> example_inputs = {input_tensor, input_tensor};
                    torch::jit::Module multi_input_script = torch::jit::trace(multi_input_module, example_inputs);
                    multi_input_script.save("multi_input_module.pt");
                    std::remove("multi_input_module.pt");
                    break;
                }
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
