#include "fuzzer_utils.h"
#include <iostream>
#include <limits>

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have more data
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, create a tensor with same shape but different values
            input2 = torch::ones_like(input1);
        }
        
        // Apply torch.fmax operation - may fail if shapes are incompatible
        try {
            torch::Tensor result = torch::fmax(input1, input2);
        } catch (...) {
            // Shape mismatch during broadcasting is expected
        }
        
        // Try element-wise variant with scalar tensor
        if (offset < Size) {
            uint8_t scalar_byte = Data[offset++];
            double scalar_value = static_cast<double>(scalar_byte) / 255.0 * 100.0;
            torch::Tensor scalar_tensor = torch::scalar_tensor(scalar_value);
            
            // Apply torch.fmax with scalar tensor (always broadcastable)
            torch::Tensor scalar_result = torch::fmax(input1, scalar_tensor);
        }
        
        // Try broadcasting with different shapes
        if (offset < Size && Size - offset > 2) {
            uint8_t rank_byte = Data[offset++];
            uint8_t rank = fuzzer_utils::parseRank(rank_byte);
            
            if (rank > 0) {
                std::vector<int64_t> new_shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
                torch::Tensor broadcast_tensor = torch::ones(new_shape);
                
                // Try broadcasting - may fail if shapes are incompatible
                try {
                    torch::Tensor broadcast_result = torch::fmax(input1, broadcast_tensor);
                } catch (...) {
                    // Broadcasting failure is expected for incompatible shapes
                }
            }
        }
        
        // Try with empty tensors
        torch::Tensor empty_tensor = torch::empty({0});
        try {
            torch::Tensor empty_result = torch::fmax(empty_tensor, empty_tensor);
        } catch (...) {
            // Expected exception for empty tensors
        }
        
        // Try with special float values using input1's shape
        // NaN values - fmax propagates non-NaN values
        torch::Tensor nan_tensor = torch::full_like(input1, std::numeric_limits<float>::quiet_NaN());
        torch::Tensor nan_result = torch::fmax(input1, nan_tensor);
        
        // Positive infinity
        torch::Tensor inf_tensor = torch::full_like(input1, std::numeric_limits<float>::infinity());
        torch::Tensor inf_result = torch::fmax(input1, inf_tensor);
        
        // Negative infinity
        torch::Tensor neg_inf_tensor = torch::full_like(input1, -std::numeric_limits<float>::infinity());
        torch::Tensor neg_inf_result = torch::fmax(input1, neg_inf_tensor);
        
        // Test with different dtypes if we have more data
        if (offset < Size) {
            uint8_t dtype_byte = Data[offset++];
            torch::ScalarType dtype;
            switch (dtype_byte % 4) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
            }
            
            try {
                torch::Tensor typed_input1 = input1.to(dtype);
                torch::Tensor typed_input2 = input2.to(dtype);
                torch::Tensor typed_result = torch::fmax(typed_input1, typed_input2);
            } catch (...) {
                // Some dtype combinations may not be supported
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}