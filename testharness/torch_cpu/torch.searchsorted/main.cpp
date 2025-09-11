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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create sorted sequence tensor
        torch::Tensor sorted_sequence = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If we've consumed all data, return
        if (offset >= Size) {
            return 0;
        }
        
        // Create values to search for
        torch::Tensor values = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract additional parameters if data available
        bool out_int32 = false;
        bool right = false;
        
        if (offset + 1 < Size) {
            out_int32 = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            right = Data[offset++] & 0x1;
        }
        
        // Try to sort the sequence tensor along the last dimension
        try {
            if (sorted_sequence.dim() > 0) {
                int64_t last_dim = sorted_sequence.dim() - 1;
                auto sort_result = sorted_sequence.sort(last_dim);
                sorted_sequence = std::get<0>(sort_result);
            }
        } catch (const std::exception&) {
            // If sorting fails, we'll use the unsorted tensor
        }
        
        // Apply searchsorted operation with different parameter combinations
        try {
            // Basic searchsorted
            torch::Tensor result1 = torch::searchsorted(sorted_sequence, values);
            
            // With out_int32 parameter
            torch::Tensor result2 = torch::searchsorted(sorted_sequence, values, out_int32);
            
            // With right parameter
            torch::Tensor result3 = torch::searchsorted(sorted_sequence, values, out_int32, right);
            
            // With side parameter as string
            torch::Tensor result4 = torch::searchsorted(sorted_sequence, values, false, false, "left");
            torch::Tensor result5 = torch::searchsorted(sorted_sequence, values, false, false, "right");
            
            // With out_int32 and side parameter
            torch::Tensor result7 = torch::searchsorted(sorted_sequence, values, out_int32, false, "left");
            torch::Tensor result8 = torch::searchsorted(sorted_sequence, values, out_int32, false, "right");
        } catch (const std::exception&) {
            // Catch exceptions from searchsorted operations
        }
        
        // Try with different tensor dimensions and edge cases
        try {
            // Create a scalar tensor if possible
            torch::Tensor scalar_value;
            if (values.numel() > 0) {
                scalar_value = values.flatten()[0];
                torch::Tensor scalar_result = torch::searchsorted(sorted_sequence, scalar_value);
            }
            
            // Try with empty tensors
            torch::Tensor empty_sequence = torch::empty({0}, sorted_sequence.options());
            torch::Tensor empty_values = torch::empty({0}, values.options());
            
            torch::Tensor empty_result1 = torch::searchsorted(empty_sequence, values);
            torch::Tensor empty_result2 = torch::searchsorted(sorted_sequence, empty_values);
            torch::Tensor empty_result3 = torch::searchsorted(empty_sequence, empty_values);
        } catch (const std::exception&) {
            // Catch exceptions from edge case operations
        }
        
        // Try with different dtypes
        try {
            auto int_options = torch::TensorOptions().dtype(torch::kInt64);
            auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
            
            // Create small tensors with different dtypes
            torch::Tensor int_sequence = torch::arange(5, int_options);
            torch::Tensor float_sequence = torch::arange(5, float_options);
            
            // Convert values to different dtypes
            torch::Tensor int_values, float_values;
            try {
                int_values = values.to(torch::kInt64);
                float_values = values.to(torch::kFloat32);
            } catch (const std::exception&) {
                int_values = torch::tensor({1, 3}, int_options);
                float_values = torch::tensor({1.5, 3.5}, float_options);
            }
            
            // Test with different dtype combinations
            torch::Tensor mixed_result1 = torch::searchsorted(int_sequence, float_values);
            torch::Tensor mixed_result2 = torch::searchsorted(float_sequence, int_values);
        } catch (const std::exception&) {
            // Catch exceptions from dtype conversion operations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
