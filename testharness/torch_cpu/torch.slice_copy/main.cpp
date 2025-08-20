#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and slice parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get parameters for slice_copy operation
        // We need at least 5 more bytes for dim, start, end, step, and source tensor
        if (offset + 5 > Size) {
            return 0;
        }
        
        // Get dimension to slice along
        int64_t dim = static_cast<int64_t>(Data[offset++]);
        if (input_tensor.dim() > 0) {
            dim = dim % input_tensor.dim();
        } else {
            dim = 0; // For scalar tensors, use dim 0
        }
        
        // Get start index
        int64_t start = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&start, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get end index
        int64_t end = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&end, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get step value
        int64_t step = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&step, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            if (step == 0) step = 1; // Avoid step=0 which would cause runtime error
        }
        
        // Create source tensor for slice_copy
        torch::Tensor source_tensor;
        if (offset < Size) {
            source_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a tensor with same properties as input
            source_tensor = torch::ones_like(input_tensor);
        }
        
        // Apply slice_copy operation
        try {
            torch::Tensor result = torch::slice_copy(input_tensor, dim, start, end, step);
            
            // Use the result to prevent optimization
            if (result.numel() > 0) {
                volatile float sum = result.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected and part of testing
        }
        
        // Try with different values to increase coverage
        try {
            // Try with negative indices
            int64_t neg_start = -start;
            int64_t neg_end = -end;
            
            torch::Tensor result = torch::slice_copy(input_tensor, dim, neg_start, neg_end, step);
            
            if (result.numel() > 0) {
                volatile float sum = result.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected and part of testing
        }
        
        // Try with None for end (represented by a large number in C++)
        try {
            torch::Tensor result = torch::slice_copy(input_tensor, dim, start, std::numeric_limits<int64_t>::max(), step);
            
            if (result.numel() > 0) {
                volatile float sum = result.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected and part of testing
        }
        
        // Try with negative step
        if (step != 0) {
            try {
                torch::Tensor result = torch::slice_copy(input_tensor, dim, start, end, -step);
                
                if (result.numel() > 0) {
                    volatile float sum = result.sum().item<float>();
                    (void)sum;
                }
            } catch (const c10::Error& e) {
                // PyTorch specific exceptions are expected and part of testing
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