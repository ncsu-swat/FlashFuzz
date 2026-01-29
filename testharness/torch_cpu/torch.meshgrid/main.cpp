#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Parse number of tensors to create (1-5)
        if (offset >= Size) return 0;
        uint8_t num_tensors = (Data[offset++] % 5) + 1;
        
        // Create input tensors for meshgrid
        std::vector<torch::Tensor> tensors;
        for (uint8_t i = 0; i < num_tensors; ++i) {
            if (offset >= Size) break;
            
            // Create a tensor with 1D shape for meshgrid input
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape to 1D if not already 1D
            if (tensor.dim() != 1) {
                int64_t numel = tensor.numel();
                if (numel > 0) {
                    tensor = tensor.reshape({numel});
                } else {
                    // Handle empty tensor case - create a small 1D tensor
                    tensor = torch::ones({1}, tensor.options());
                }
            }
            
            // Limit tensor size to avoid memory issues
            if (tensor.size(0) > 100) {
                tensor = tensor.slice(0, 0, 100);
            }
            
            tensors.push_back(tensor);
        }
        
        if (tensors.empty()) {
            // Ensure we have at least one tensor
            tensors.push_back(torch::ones({1}));
        }
        
        // Parse indexing option
        bool indexing_ij = true;
        if (offset < Size) {
            indexing_ij = (Data[offset++] % 2) == 0;
        }
        
        // Apply meshgrid operation with different indexing modes
        std::vector<torch::Tensor> result;
        if (indexing_ij) {
            result = torch::meshgrid(tensors, "ij");
        } else {
            result = torch::meshgrid(tensors, "xy");
        }
        
        // Perform operations on the result to ensure code paths are exercised
        for (const auto& res_tensor : result) {
            // Exercise various tensor operations on the result
            volatile auto dim = res_tensor.dim();
            volatile auto numel = res_tensor.numel();
            
            if (res_tensor.numel() > 0) {
                // Compute sum to exercise the tensor data
                torch::Tensor sum = res_tensor.sum();
                volatile float sum_val = sum.item<float>();
                (void)sum_val;
                
                // Exercise shape operations
                auto sizes = res_tensor.sizes();
                (void)sizes;
            }
        }
        
        // Also test the deprecated version without indexing argument for coverage
        // (will use default "ij" indexing)
        try {
            auto result_default = torch::meshgrid(tensors);
            (void)result_default;
        } catch (...) {
            // Silently ignore - deprecated API may behave differently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}