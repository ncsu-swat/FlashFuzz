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
        
        // Create a tensor to test with
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test basic tensor operations since torch::overrides is not available in C++ frontend
        
        // 1. Test tensor properties
        auto sizes = tensor.sizes();
        auto dtype = tensor.dtype();
        auto device = tensor.device();
        
        // 2. Test tensor methods
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                auto result = tensor.add(tensor2);
            } catch (...) {
                // Ignore exceptions
            }
            
            try {
                auto result = tensor.mul(tensor2);
            } catch (...) {
                // Ignore exceptions
            }
        }
        
        // 3. Test tensor cloning and copying
        auto cloned = tensor.clone();
        auto detached = tensor.detach();
        
        // 4. Test tensor shape operations
        if (offset < Size) {
            uint8_t operation_selector = Data[offset++] % 5;
            
            switch (operation_selector) {
                case 0:
                    try {
                        auto reshaped = tensor.reshape({-1});
                    } catch (...) {}
                    break;
                case 1:
                    try {
                        auto flattened = tensor.flatten();
                    } catch (...) {}
                    break;
                case 2:
                    try {
                        auto squeezed = tensor.squeeze();
                    } catch (...) {}
                    break;
                case 3:
                    try {
                        auto unsqueezed = tensor.unsqueeze(0);
                    } catch (...) {}
                    break;
                case 4:
                    try {
                        auto transposed = tensor.transpose(0, -1);
                    } catch (...) {}
                    break;
            }
        }
        
        // 5. Test mathematical operations
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                auto sum_result = tensor + tensor2;
            } catch (...) {}
            
            try {
                auto sub_result = tensor - tensor2;
            } catch (...) {}
            
            try {
                auto mul_result = tensor * tensor2;
            } catch (...) {}
            
            try {
                auto div_result = tensor / (tensor2 + 1e-8);
            } catch (...) {}
        }
        
        // 6. Test unary operations
        if (offset < Size) {
            uint8_t unary_selector = Data[offset++] % 5;
            
            switch (unary_selector) {
                case 0:
                    try {
                        auto sin_result = tensor.sin();
                    } catch (...) {}
                    break;
                case 1:
                    try {
                        auto cos_result = tensor.cos();
                    } catch (...) {}
                    break;
                case 2:
                    try {
                        auto exp_result = tensor.exp();
                    } catch (...) {}
                    break;
                case 3:
                    try {
                        auto log_result = tensor.log();
                    } catch (...) {}
                    break;
                case 4:
                    try {
                        auto abs_result = tensor.abs();
                    } catch (...) {}
                    break;
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
