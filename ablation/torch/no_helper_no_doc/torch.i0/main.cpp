#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various shapes and dtypes
        auto input_tensor = generateTensor(Data, Size, offset);
        
        // Test basic i0 operation
        auto result = torch::i0(input_tensor);
        
        // Test with different tensor properties
        if (offset < Size) {
            // Test with different dtypes if possible
            auto float_tensor = input_tensor.to(torch::kFloat32);
            auto float_result = torch::i0(float_tensor);
            
            auto double_tensor = input_tensor.to(torch::kFloat64);
            auto double_result = torch::i0(double_tensor);
        }
        
        // Test with special values if we have enough data
        if (offset + sizeof(float) * 4 <= Size) {
            // Create tensor with special values
            std::vector<float> special_values;
            
            // Add some regular values
            float val1, val2, val3, val4;
            memcpy(&val1, Data + offset, sizeof(float)); offset += sizeof(float);
            memcpy(&val2, Data + offset, sizeof(float)); offset += sizeof(float);
            memcpy(&val3, Data + offset, sizeof(float)); offset += sizeof(float);
            memcpy(&val4, Data + offset, sizeof(float)); offset += sizeof(float);
            
            special_values.push_back(val1);
            special_values.push_back(val2);
            special_values.push_back(val3);
            special_values.push_back(val4);
            
            // Add some edge case values
            special_values.push_back(0.0f);
            special_values.push_back(-0.0f);
            special_values.push_back(1.0f);
            special_values.push_back(-1.0f);
            special_values.push_back(std::numeric_limits<float>::infinity());
            special_values.push_back(-std::numeric_limits<float>::infinity());
            special_values.push_back(std::numeric_limits<float>::quiet_NaN());
            special_values.push_back(std::numeric_limits<float>::max());
            special_values.push_back(std::numeric_limits<float>::lowest());
            special_values.push_back(std::numeric_limits<float>::min());
            special_values.push_back(std::numeric_limits<float>::denorm_min());
            
            auto special_tensor = torch::tensor(special_values);
            auto special_result = torch::i0(special_tensor);
        }
        
        // Test with different tensor shapes
        if (input_tensor.numel() > 0) {
            // Test with reshaped tensor
            auto flat_tensor = input_tensor.flatten();
            auto flat_result = torch::i0(flat_tensor);
            
            // Test with squeezed/unsqueezed tensors
            auto unsqueezed = input_tensor.unsqueeze(0);
            auto unsqueezed_result = torch::i0(unsqueezed);
            
            if (input_tensor.dim() > 0) {
                auto squeezed = input_tensor.squeeze();
                auto squeezed_result = torch::i0(squeezed);
            }
        }
        
        // Test with contiguous and non-contiguous tensors
        if (input_tensor.dim() >= 2) {
            auto transposed = input_tensor.transpose(0, 1);
            auto transposed_result = torch::i0(transposed);
            
            // Make it contiguous and test again
            auto contiguous_transposed = transposed.contiguous();
            auto contiguous_result = torch::i0(contiguous_transposed);
        }
        
        // Test with tensors on different devices if CUDA is available
        if (torch::cuda::is_available() && input_tensor.numel() > 0) {
            auto cuda_tensor = input_tensor.to(torch::kCUDA);
            auto cuda_result = torch::i0(cuda_tensor);
        }
        
        // Test with requires_grad
        if (input_tensor.dtype().is_floating_point()) {
            auto grad_tensor = input_tensor.clone().requires_grad_(true);
            auto grad_result = torch::i0(grad_tensor);
            
            // Test backward pass if result is scalar or we can sum it
            if (grad_result.numel() == 1) {
                grad_result.backward();
            } else if (grad_result.numel() > 0) {
                grad_result.sum().backward();
            }
        }
        
        // Test with very large and very small values
        if (offset + sizeof(int) <= Size) {
            int scale_factor;
            memcpy(&scale_factor, Data + offset, sizeof(int));
            offset += sizeof(int);
            
            // Create scaled versions
            auto large_tensor = input_tensor * std::abs(scale_factor % 100 + 1);
            auto large_result = torch::i0(large_tensor);
            
            auto small_tensor = input_tensor / std::max(1, std::abs(scale_factor % 100 + 1));
            auto small_result = torch::i0(small_tensor);
        }
        
        // Verify result properties
        if (result.defined()) {
            // i0 should preserve input shape
            if (result.sizes() != input_tensor.sizes()) {
                std::cerr << "Shape mismatch in i0 result" << std::endl;
            }
            
            // Result should be real-valued for real input
            if (input_tensor.dtype().is_floating_point() && !result.dtype().is_floating_point()) {
                std::cerr << "Unexpected dtype in i0 result" << std::endl;
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}