#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Check if deterministic algorithms are enabled
        bool are_enabled = torch::are_deterministic_algorithms_enabled();
        
        // Toggle deterministic algorithms
        if (Size > offset)
        {
            bool should_enable = Data[offset++] % 2 == 0;
            torch::use_deterministic_algorithms(should_enable);
            
            // Verify the setting was applied
            bool new_state = torch::are_deterministic_algorithms_enabled();
            
            // Create a tensor to test with deterministic operations
            if (Size > offset)
            {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Perform operations that might be affected by deterministic mode
                if (tensor.defined() && tensor.numel() > 0)
                {
                    // Operations that might be affected by deterministic algorithms
                    torch::Tensor result;
                    
                    if (tensor.dim() > 0)
                    {
                        // Max pooling is affected by deterministic algorithms
                        if (tensor.dim() >= 2 && tensor.size(0) > 0)
                        {
                            try {
                                // Reshape tensor to have at least 3 dimensions for max_pool2d
                                auto input = tensor;
                                if (tensor.dim() == 1)
                                {
                                    input = tensor.reshape({1, tensor.size(0), 1});
                                }
                                else if (tensor.dim() == 2)
                                {
                                    input = tensor.unsqueeze(0);
                                }
                                
                                // Ensure input has float dtype for max_pool2d
                                if (input.scalar_type() != torch::kFloat && 
                                    input.scalar_type() != torch::kDouble && 
                                    input.scalar_type() != torch::kHalf)
                                {
                                    input = input.to(torch::kFloat);
                                }
                                
                                // Apply max pooling which is affected by deterministic mode
                                result = torch::max_pool2d(input, {2, 2}, {1, 1}, {0, 0}, {1, 1}, false);
                            }
                            catch (const std::exception&) {
                                // Ignore exceptions from max_pool2d
                            }
                        }
                    }
                    
                    // Test other operations that might be affected by deterministic mode
                    try {
                        result = torch::conv2d(tensor.to(torch::kFloat).reshape({1, 1, tensor.numel(), 1}), 
                                              torch::ones({1, 1, 3, 3}));
                    }
                    catch (const std::exception&) {
                        // Ignore exceptions from conv2d
                    }
                }
            }
            
            // Reset to original state
            torch::use_deterministic_algorithms(are_enabled);
        }
        
        // Test with different CUDA settings if available
        if (torch::cuda::is_available() && Size > offset)
        {
            bool use_cuda = Data[offset++] % 2 == 0;
            
            if (use_cuda)
            {
                torch::Device device(torch::kCUDA);
                
                // Test CUDA-specific deterministic behavior
                bool cuda_deterministic = torch::backends::cudnn::deterministic();
                
                // Toggle CUDA deterministic mode
                if (Size > offset)
                {
                    bool should_be_deterministic = Data[offset++] % 2 == 0;
                    torch::backends::cudnn::set_deterministic(should_be_deterministic);
                    
                    // Verify the setting was applied
                    bool new_cuda_deterministic = torch::backends::cudnn::deterministic();
                    
                    // Create a tensor on CUDA to test with deterministic operations
                    if (Size > offset)
                    {
                        try {
                            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                            if (tensor.defined())
                            {
                                tensor = tensor.to(device);
                                
                                // Perform operations that might be affected by deterministic mode
                                if (tensor.numel() > 0)
                                {
                                    try {
                                        // Operations affected by CUDA deterministic mode
                                        if (tensor.dim() >= 2)
                                        {
                                            auto input = tensor;
                                            if (input.scalar_type() != torch::kFloat && 
                                                input.scalar_type() != torch::kDouble && 
                                                input.scalar_type() != torch::kHalf)
                                            {
                                                input = input.to(torch::kFloat);
                                            }
                                            
                                            if (input.dim() == 2)
                                            {
                                                input = input.unsqueeze(0).unsqueeze(0);
                                            }
                                            
                                            torch::Tensor result = torch::max_pool2d(input, {2, 2});
                                        }
                                    }
                                    catch (const std::exception&) {
                                        // Ignore exceptions from operations
                                    }
                                }
                            }
                        }
                        catch (const std::exception&) {
                            // Ignore exceptions from tensor creation
                        }
                    }
                    
                    // Reset to original state
                    torch::backends::cudnn::set_deterministic(cuda_deterministic);
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