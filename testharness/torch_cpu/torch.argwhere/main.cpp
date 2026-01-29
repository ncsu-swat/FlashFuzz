#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.argwhere operation
        torch::Tensor result = torch::argwhere(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto sizes = result.sizes();
            
            // Force evaluation by accessing data
            // argwhere returns int64 tensor of shape (num_nonzero, input.dim())
            auto result_cpu = result.cpu().contiguous();
            volatile int64_t first_element = result_cpu.data_ptr<int64_t>()[0];
            (void)first_element;
        }
        
        // Try with different options if we have more data
        if (Size - offset >= 1 && offset < Size) {
            uint8_t option_byte = Data[offset++];
            (void)option_byte;
            
            // Create a boolean mask from the original tensor
            try {
                torch::Tensor bool_mask = input_tensor.to(torch::kBool);
                
                // Apply argwhere on the boolean mask
                torch::Tensor bool_result = torch::argwhere(bool_mask);
                
                if (bool_result.defined() && bool_result.numel() > 0) {
                    volatile int64_t val = bool_result.cpu().contiguous().data_ptr<int64_t>()[0];
                    (void)val;
                }
            } catch (...) {
                // Silently ignore conversion failures
            }
            
            // Try with a tensor containing NaN values if the tensor is floating point
            if (at::isFloatingType(input_tensor.scalar_type())) {
                try {
                    torch::Tensor nan_tensor = input_tensor.clone();
                    
                    // Insert some NaN values if tensor is not empty
                    if (nan_tensor.numel() > 0) {
                        // Set first element to NaN using fill with a slice
                        auto flat = nan_tensor.flatten();
                        flat.index_put_({0}, std::numeric_limits<float>::quiet_NaN());
                        
                        // Apply argwhere on tensor with NaN
                        torch::Tensor nan_result = torch::argwhere(nan_tensor);
                        
                        if (nan_result.defined() && nan_result.numel() > 0) {
                            volatile int64_t val = nan_result.cpu().contiguous().data_ptr<int64_t>()[0];
                            (void)val;
                        }
                    }
                } catch (...) {
                    // Silently ignore failures with NaN handling
                }
            }
            
            // Try with a tensor containing infinity values if the tensor is floating point
            if (at::isFloatingType(input_tensor.scalar_type())) {
                try {
                    torch::Tensor inf_tensor = input_tensor.clone();
                    
                    // Insert some infinity values if tensor is not empty
                    if (inf_tensor.numel() > 0) {
                        // Set first element to infinity
                        auto flat = inf_tensor.flatten();
                        flat.index_put_({0}, std::numeric_limits<float>::infinity());
                        
                        // Apply argwhere on tensor with infinity
                        torch::Tensor inf_result = torch::argwhere(inf_tensor);
                        
                        if (inf_result.defined() && inf_result.numel() > 0) {
                            volatile int64_t val = inf_result.cpu().contiguous().data_ptr<int64_t>()[0];
                            (void)val;
                        }
                    }
                } catch (...) {
                    // Silently ignore failures with infinity handling
                }
            }
            
            // Try with zero tensor (should return empty result)
            try {
                torch::Tensor zero_tensor = torch::zeros_like(input_tensor);
                torch::Tensor zero_result = torch::argwhere(zero_tensor);
                
                // Result should be empty for all-zero tensor
                volatile int64_t zero_numel = zero_result.numel();
                (void)zero_numel;
            } catch (...) {
                // Silently ignore
            }
            
            // Try with all-ones tensor (should return all indices)
            try {
                torch::Tensor ones_tensor = torch::ones_like(input_tensor);
                torch::Tensor ones_result = torch::argwhere(ones_tensor);
                
                if (ones_result.defined() && ones_result.numel() > 0) {
                    volatile int64_t val = ones_result.cpu().contiguous().data_ptr<int64_t>()[0];
                    (void)val;
                }
            } catch (...) {
                // Silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}