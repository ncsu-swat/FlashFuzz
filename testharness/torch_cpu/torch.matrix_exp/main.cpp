#include "fuzzer_utils.h"
#include <iostream>

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
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Use first byte to determine matrix size (1-8)
        int64_t matrix_size = (Data[0] % 8) + 1;
        offset = 1;
        
        // Use second byte to determine dtype
        uint8_t dtype_selector = Data[1] % 3;
        offset = 2;
        
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kComplexFloat; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Calculate number of elements needed for square matrix
        int64_t num_elements = matrix_size * matrix_size;
        
        // Check if we have enough data
        size_t bytes_needed = num_elements * sizeof(float);
        if (Size - offset < bytes_needed) {
            // Fall back to smaller matrix
            matrix_size = 2;
            num_elements = 4;
        }
        
        // Create a tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        
        // Ensure we have a proper square matrix
        torch::Tensor square_matrix;
        
        if (input.numel() >= num_elements) {
            // Flatten and take first num_elements, then reshape to square
            square_matrix = input.flatten().slice(0, 0, num_elements).reshape({matrix_size, matrix_size});
        } else if (input.numel() > 0) {
            // Use what we have and pad with zeros
            int64_t available = input.numel();
            int64_t side = static_cast<int64_t>(std::sqrt(static_cast<double>(available)));
            if (side < 1) side = 1;
            int64_t needed = side * side;
            square_matrix = input.flatten().slice(0, 0, std::min(available, needed));
            if (square_matrix.numel() < needed) {
                square_matrix = torch::cat({square_matrix, torch::zeros(needed - square_matrix.numel(), input.options())});
            }
            square_matrix = square_matrix.reshape({side, side});
        } else {
            // Create a small random matrix
            square_matrix = torch::randn({2, 2});
        }
        
        // Convert to target dtype
        try {
            square_matrix = square_matrix.to(dtype);
        } catch (...) {
            // If dtype conversion fails, use float32
            square_matrix = square_matrix.to(torch::kFloat32);
        }
        
        // Apply matrix_exp
        torch::Tensor result = torch::matrix_exp(square_matrix);
        
        // Test with batched input if we have enough data
        if (Size > 32) {
            uint8_t batch_size = (Data[Size - 1] % 3) + 1;
            int64_t small_size = 2;
            
            // Create batched square matrices
            torch::Tensor batched = torch::randn({batch_size, small_size, small_size}, 
                                                  torch::TensorOptions().dtype(torch::kFloat32));
            
            // Fill with some fuzzer-derived values
            auto accessor = batched.accessor<float, 3>();
            size_t data_idx = offset;
            for (int64_t b = 0; b < batch_size && data_idx < Size; ++b) {
                for (int64_t i = 0; i < small_size; ++i) {
                    for (int64_t j = 0; j < small_size && data_idx < Size; ++j) {
                        // Scale to reasonable range to avoid overflow
                        accessor[b][i][j] = static_cast<float>(Data[data_idx++] - 128) / 32.0f;
                    }
                }
            }
            
            torch::Tensor batched_result = torch::matrix_exp(batched);
        }
        
        // Test with complex input
        if (Size > 16) {
            torch::Tensor complex_input = torch::randn({2, 2}, torch::kComplexFloat);
            // Fill with fuzzer data
            if (Size - offset >= 8) {
                float* data_ptr = reinterpret_cast<float*>(complex_input.data_ptr());
                for (int i = 0; i < 8 && offset + i * 4 + 3 < Size; ++i) {
                    uint32_t val = (Data[offset + i * 4] << 24) | 
                                   (Data[offset + i * 4 + 1] << 16) |
                                   (Data[offset + i * 4 + 2] << 8) | 
                                   Data[offset + i * 4 + 3];
                    data_ptr[i] = static_cast<float>(val) / static_cast<float>(UINT32_MAX) * 2.0f - 1.0f;
                }
            }
            
            try {
                torch::Tensor complex_result = torch::matrix_exp(complex_input);
            } catch (...) {
                // Complex operations may fail on some platforms
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