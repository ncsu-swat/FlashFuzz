#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse reduction type from the next byte if available
        std::string reduction_type = "sum";
        if (offset < Size) {
            uint8_t reduction_selector = Data[offset++];
            switch (reduction_selector % 4) {
                case 0: reduction_type = "sum"; break;
                case 1: reduction_type = "mean"; break;
                case 2: reduction_type = "max"; break;
                case 3: reduction_type = "min"; break;
            }
        }
        
        // Create segment lengths tensor
        torch::Tensor lengths;
        if (offset < Size) {
            lengths = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to ensure lengths is 1D and contains non-negative integers
            if (lengths.dim() > 1) {
                lengths = lengths.flatten();
            }
            
            // Convert to int64 if not already
            if (lengths.scalar_type() != torch::kInt64) {
                lengths = lengths.to(torch::kInt64);
            }
        } else {
            // Default lengths if we don't have enough data
            if (input.dim() > 0) {
                int64_t first_dim = input.size(0);
                lengths = torch::ones({first_dim}, torch::kInt64);
            } else {
                lengths = torch::ones({1}, torch::kInt64);
            }
        }
        
        // Parse axis from the next byte if available
        int64_t axis = 0;
        if (offset < Size) {
            uint8_t dim_byte = Data[offset++];
            if (input.dim() > 0) {
                axis = dim_byte % std::max(1L, static_cast<int64_t>(input.dim()));
            }
        }
        
        // Try different segment_reduce operations
        try {
            torch::Tensor result = torch::segment_reduce(input, reduction_type, lengths, torch::nullopt, torch::nullopt, axis);
        } catch (const c10::Error &e) {
            // Expected exceptions from PyTorch operations are fine
        }
        
        // Try with unsafe=true option
        try {
            torch::Tensor result_unsafe = torch::segment_reduce(input, reduction_type, lengths, torch::nullopt, torch::nullopt, axis, true);
        } catch (const c10::Error &e) {
            // Expected exceptions from PyTorch operations are fine
        }
        
        // Try with different axis if input has multiple dimensions
        if (input.dim() > 1 && offset < Size) {
            int64_t new_axis = Data[offset++] % input.dim();
            try {
                torch::Tensor result_axis = torch::segment_reduce(input, reduction_type, lengths, torch::nullopt, torch::nullopt, new_axis);
            } catch (const c10::Error &e) {
                // Expected exceptions from PyTorch operations are fine
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