#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <ATen/Version.h> // For version functions

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Get the PyTorch version information using available functions
        auto mkl_version = torch::get_mkl_version();
        auto mkldnn_version = torch::get_mkldnn_version();
        
        // Create a tensor with some version-related data
        if (Size > 0) {
            // Create a tensor with some numeric data
            std::vector<int64_t> version_data = {1, 0, 0};
            auto version_tensor = torch::tensor(version_data);
            
            // Try to create a tensor from the fuzzer data
            if (Size > 2) {
                auto input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Perform some operations with the version information
                if (input_tensor.defined() && input_tensor.numel() > 0) {
                    // Try to use version information with the tensor
                    if (input_tensor.scalar_type() == torch::kInt64 || 
                        input_tensor.scalar_type() == torch::kInt32 || 
                        input_tensor.scalar_type() == torch::kFloat) {
                        
                        // Create a tensor with the same shape as input but filled with a constant
                        auto version_filled = torch::full_like(input_tensor, 1.0f);
                        
                        // Try some operations
                        auto result = version_filled + input_tensor;
                        
                        // Check if cuda is available
                        if (torch::cuda::is_available()) {
                            auto cuda_tensor = torch::tensor({1});
                        }
                    }
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