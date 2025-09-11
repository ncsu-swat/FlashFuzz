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
        
        // Create a module with a parameter
        torch::nn::Linear module(10, 5);
        
        // Create a tensor to use as a parametrization
        torch::Tensor param_tensor;
        try {
            param_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception&) {
            // If tensor creation fails, create a simple tensor
            param_tensor = torch::ones({10, 5});
        }
        
        // Try different operations based on remaining data
        if (offset < Size) {
            uint8_t op_type = Data[offset++] % 4;
            
            switch (op_type) {
                case 0: {
                    // Test basic parameter access
                    try {
                        auto params = module->named_parameters();
                        for (auto& param : params) {
                            torch::Tensor modified = param.value() * param_tensor;
                        }
                    } catch (const std::exception&) {
                        // Ignore exceptions
                    }
                    break;
                }
                case 1: {
                    // Test parameter modification
                    try {
                        auto params = module->named_parameters();
                        for (auto& param : params) {
                            torch::Tensor modified = param.value() + param_tensor;
                        }
                    } catch (const std::exception&) {
                        // Ignore exceptions
                    }
                    break;
                }
                case 2: {
                    // Test parameter transformation
                    try {
                        auto params = module->named_parameters();
                        for (auto& param : params) {
                            torch::Tensor modified = torch::sigmoid(param.value()) * param_tensor;
                        }
                    } catch (const std::exception&) {
                        // Ignore exceptions
                    }
                    break;
                }
                case 3: {
                    // Test multiple parameter operations
                    try {
                        auto params = module->named_parameters();
                        for (auto& param : params) {
                            torch::Tensor temp = param.value() * 2;
                            torch::Tensor modified = temp + param_tensor;
                        }
                    } catch (const std::exception&) {
                        // Ignore exceptions
                    }
                    break;
                }
            }
        } else {
            // Default case if no more data
            try {
                auto params = module->named_parameters();
                for (auto& param : params) {
                    torch::Tensor modified = param.value() * 0.5;
                }
            } catch (const std::exception&) {
                // Ignore exceptions
            }
        }
        
        // Test the module
        torch::Tensor input;
        try {
            if (offset < Size) {
                input = fuzzer_utils::createTensor(Data, Size, offset);
                if (input.sizes().size() < 2 || input.sizes()[0] != 10) {
                    input = torch::ones({10, 10});
                }
            } else {
                input = torch::ones({10, 10});
            }
            
            torch::Tensor output = module->forward(input);
        } catch (const std::exception&) {
            // Ignore exceptions from forward pass
        }
        
        // Test parameter-specific operations
        if (offset < Size && Data[offset++] % 2 == 0) {
            try {
                // Get parameters
                auto params = module->named_parameters();
                for (auto& param : params) {
                    torch::Tensor original = param.value();
                    torch::Tensor modified = original.clone();
                }
            } catch (const std::exception&) {
                // Ignore exceptions
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
