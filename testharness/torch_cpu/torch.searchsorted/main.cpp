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
            out_int32 = (Data[offset++] & 0x1) != 0;
        }
        
        if (offset < Size) {
            right = (Data[offset++] & 0x1) != 0;
        }
        
        // Ensure tensors are floating point for sort operation
        try {
            if (!sorted_sequence.is_floating_point()) {
                sorted_sequence = sorted_sequence.to(torch::kFloat32);
            }
            if (!values.is_floating_point()) {
                values = values.to(torch::kFloat32);
            }
        } catch (const std::exception&) {
            return 0;
        }
        
        // Try to sort the sequence tensor along the last dimension
        try {
            if (sorted_sequence.dim() > 0 && sorted_sequence.numel() > 0) {
                int64_t last_dim = sorted_sequence.dim() - 1;
                auto sort_result = sorted_sequence.sort(last_dim);
                sorted_sequence = std::get<0>(sort_result);
            }
        } catch (const std::exception&) {
            // If sorting fails, continue with unsorted tensor
        }
        
        // Apply searchsorted operation with different parameter combinations
        try {
            // Basic searchsorted - using the correct C++ API
            // torch::searchsorted(sorted_sequence, values, out_int32, right, side, sorter)
            torch::Tensor result1 = torch::searchsorted(sorted_sequence, values);
            
            // With out_int32 parameter
            torch::Tensor result2 = torch::searchsorted(sorted_sequence, values, out_int32);
            
            // With out_int32 and right parameters
            torch::Tensor result3 = torch::searchsorted(sorted_sequence, values, out_int32, right);
        } catch (const std::exception&) {
            // Catch exceptions from searchsorted operations (shape mismatches, etc.)
        }
        
        // Try with side parameter
        try {
            torch::Tensor result_left = torch::searchsorted(sorted_sequence, values, false, false, "left");
            torch::Tensor result_right = torch::searchsorted(sorted_sequence, values, false, false, "right");
        } catch (const std::exception&) {
            // Catch exceptions from side parameter variations
        }
        
        // Try with out_int32 variations and side
        try {
            torch::Tensor result_int32_left = torch::searchsorted(sorted_sequence, values, true, false, "left");
            torch::Tensor result_int32_right = torch::searchsorted(sorted_sequence, values, true, false, "right");
        } catch (const std::exception&) {
            // Catch exceptions
        }
        
        // Try with 1D tensors explicitly (common use case)
        try {
            torch::Tensor flat_sequence = sorted_sequence.flatten();
            torch::Tensor flat_values = values.flatten();
            
            // Sort the flattened sequence
            if (flat_sequence.numel() > 0) {
                auto sorted = flat_sequence.sort(0);
                flat_sequence = std::get<0>(sorted);
                
                torch::Tensor flat_result = torch::searchsorted(flat_sequence, flat_values);
                torch::Tensor flat_result_right = torch::searchsorted(flat_sequence, flat_values, false, true);
            }
        } catch (const std::exception&) {
            // Catch exceptions from flattened operations
        }
        
        // Try with scalar values
        try {
            if (values.numel() > 0) {
                torch::Tensor scalar_value = values.flatten().index({0});
                if (sorted_sequence.dim() == 1 && sorted_sequence.numel() > 0) {
                    torch::Tensor scalar_result = torch::searchsorted(sorted_sequence, scalar_value);
                }
            }
        } catch (const std::exception&) {
            // Catch exceptions from scalar operations
        }
        
        // Try with different dtypes
        try {
            auto int_options = torch::TensorOptions().dtype(torch::kInt64);
            auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
            auto double_options = torch::TensorOptions().dtype(torch::kFloat64);
            
            // Create small sorted tensors with different dtypes
            torch::Tensor int_sequence = torch::arange(10, int_options);
            torch::Tensor float_sequence = torch::linspace(0, 10, 10, float_options);
            torch::Tensor double_sequence = torch::linspace(0, 10, 10, double_options);
            
            // Create test values
            torch::Tensor test_values_float = torch::tensor({1.5, 3.5, 7.5}, float_options);
            torch::Tensor test_values_int = torch::tensor({1, 3, 7}, int_options);
            
            // Test with different dtype combinations
            torch::Tensor result_int_float = torch::searchsorted(int_sequence, test_values_float);
            torch::Tensor result_float_int = torch::searchsorted(float_sequence, test_values_int);
            torch::Tensor result_double = torch::searchsorted(double_sequence, test_values_float);
        } catch (const std::exception&) {
            // Catch exceptions from dtype operations
        }
        
        // Try with batched input (2D sorted_sequence)
        try {
            if (sorted_sequence.dim() >= 1 && sorted_sequence.numel() > 0) {
                // Create a 2D batch of sorted sequences
                torch::Tensor batch_sequence = sorted_sequence.unsqueeze(0).expand({3, -1});
                if (batch_sequence.dim() == 2) {
                    // Sort along last dimension
                    auto sorted_batch = batch_sequence.sort(-1);
                    batch_sequence = std::get<0>(sorted_batch);
                    
                    torch::Tensor batch_values = values.flatten();
                    if (batch_values.numel() > 0) {
                        torch::Tensor batch_result = torch::searchsorted(batch_sequence, batch_values);
                    }
                }
            }
        } catch (const std::exception&) {
            // Catch exceptions from batched operations
        }
        
        // Try with sorter argument (optional tensor providing sort indices)
        try {
            if (sorted_sequence.dim() == 1 && sorted_sequence.numel() > 1) {
                // Create unsorted sequence and corresponding sorter
                torch::Tensor unsorted = sorted_sequence.clone();
                auto sort_result = unsorted.sort(0);
                torch::Tensor sorter = std::get<1>(sort_result);
                
                // Use sorter to indicate sort order without actually sorting
                torch::Tensor result_with_sorter = torch::searchsorted(
                    unsorted, values.flatten(), false, false, c10::nullopt, sorter);
            }
        } catch (const std::exception&) {
            // Catch exceptions from sorter operations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}