#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
        
        // Apply meshgrid operation
        std::vector<torch::Tensor> result;
        if (indexing_ij) {
            result = torch::meshgrid(tensors, "ij");
        } else {
            result = torch::meshgrid(tensors, "xy");
        }
        
        // Verify result
        if (result.size() != tensors.size()) {
            throw std::runtime_error("Unexpected result size from meshgrid");
        }
        
        // Perform some operations on the result to ensure it's used
        for (auto& res_tensor : result) {
            torch::Tensor sum = res_tensor.sum();
            if (sum.numel() > 0) {
                float sum_val = sum.item<float>();
                // Use sum_val to prevent compiler optimization
                if (std::isnan(sum_val)) {
                    throw std::runtime_error("NaN detected in result");
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