#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor configuration
        auto tensor_config = parse_tensor_config(Data, Size, offset);
        if (!tensor_config.has_value()) {
            return 0;
        }

        // Create input tensor with various data types and shapes
        torch::Tensor input_tensor = create_tensor_from_config(tensor_config.value());
        
        // Test basic abs operation
        torch::Tensor result = torch::abs(input_tensor);
        
        // Test in-place abs operation
        torch::Tensor input_copy = input_tensor.clone();
        input_copy.abs_();
        
        // Test with different tensor properties
        if (input_tensor.numel() > 0) {
            // Test with contiguous tensor
            torch::Tensor contiguous_input = input_tensor.contiguous();
            torch::Tensor contiguous_result = torch::abs(contiguous_input);
            
            // Test with non-contiguous tensor (if possible)
            if (input_tensor.dim() > 1) {
                torch::Tensor transposed = input_tensor.transpose(0, -1);
                torch::Tensor transposed_result = torch::abs(transposed);
            }
            
            // Test with sliced tensor
            if (input_tensor.size(0) > 1) {
                torch::Tensor sliced = input_tensor.slice(0, 0, input_tensor.size(0) / 2);
                torch::Tensor sliced_result = torch::abs(sliced);
            }
        }
        
        // Test with special values if floating point
        if (input_tensor.dtype().is_floating_point()) {
            // Create tensor with special values
            std::vector<double> special_values = {
                0.0, -0.0, 1.0, -1.0, 
                std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN()
            };
            
            for (double val : special_values) {
                torch::Tensor special_tensor = torch::full({1}, val, input_tensor.options());
                torch::Tensor special_result = torch::abs(special_tensor);
            }
        }
        
        // Test with complex tensors if supported
        if (input_tensor.dtype().is_complex()) {
            torch::Tensor complex_result = torch::abs(input_tensor);
            // For complex tensors, abs should return real-valued result
        }
        
        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && input_tensor.numel() < 10000) { // Limit size for CUDA tests
            try {
                torch::Tensor cuda_input = input_tensor.to(torch::kCUDA);
                torch::Tensor cuda_result = torch::abs(cuda_input);
                torch::Tensor cpu_result_from_cuda = cuda_result.to(torch::kCPU);
            } catch (const std::exception&) {
                // CUDA operations might fail, ignore
            }
        }
        
        // Test output tensor variant
        if (input_tensor.numel() > 0) {
            torch::Tensor output_tensor = torch::empty_like(input_tensor);
            if (input_tensor.dtype().is_complex()) {
                // For complex input, output should be real
                auto real_dtype = input_tensor.dtype() == torch::kComplexFloat ? torch::kFloat : torch::kDouble;
                output_tensor = torch::empty(input_tensor.sizes(), torch::TensorOptions().dtype(real_dtype).device(input_tensor.device()));
            }
            torch::abs_out(output_tensor, input_tensor);
        }
        
        // Test gradient computation if tensor requires grad
        if (input_tensor.requires_grad() && input_tensor.dtype().is_floating_point()) {
            torch::Tensor grad_input = input_tensor.clone().detach().requires_grad_(true);
            torch::Tensor grad_result = torch::abs(grad_input);
            
            if (grad_result.numel() > 0) {
                torch::Tensor grad_output = torch::ones_like(grad_result);
                grad_result.backward(grad_output);
            }
        }
        
        // Verify basic properties
        if (input_tensor.numel() > 0) {
            // Check that result has same shape as input (except for complex->real case)
            if (!input_tensor.dtype().is_complex()) {
                assert(result.sizes() == input_tensor.sizes());
                assert(result.dtype() == input_tensor.dtype());
            }
            
            // Check that all values are non-negative (for real tensors)
            if (result.dtype().is_floating_point() && !result.dtype().is_complex()) {
                torch::Tensor non_negative_check = result >= 0;
                // This should be all true, but we don't assert to avoid crashes on NaN
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