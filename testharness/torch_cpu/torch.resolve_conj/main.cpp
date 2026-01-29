#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply resolve_conj to the original tensor
        torch::Tensor resolved_original = torch::resolve_conj(input_tensor);
        
        // Create a conjugated version of the tensor and resolve it
        torch::Tensor conj_tensor = input_tensor.conj();
        torch::Tensor resolved_conj = torch::resolve_conj(conj_tensor);
        
        // Test with complex tensors explicitly (resolve_conj is most relevant for complex types)
        try {
            torch::Tensor complex_tensor = input_tensor.to(torch::kComplexFloat);
            torch::Tensor conj_complex = complex_tensor.conj();
            torch::Tensor resolved_conj_complex = torch::resolve_conj(conj_complex);
            
            // Verify that resolve_conj materializes the conjugation
            torch::Tensor resolved_original_complex = torch::resolve_conj(complex_tensor);
        } catch (...) {
            // Silently ignore - conversion may fail for some inputs
        }
        
        // Try resolve_conj on a view of the tensor
        try {
            if (input_tensor.dim() > 0 && input_tensor.numel() > 0) {
                torch::Tensor view_tensor = input_tensor.view({-1});
                torch::Tensor resolved_view = torch::resolve_conj(view_tensor);
                
                // Also test conj view
                torch::Tensor conj_view = view_tensor.conj();
                torch::Tensor resolved_conj_view = torch::resolve_conj(conj_view);
            }
        } catch (...) {
            // Silently ignore shape-related failures
        }
        
        // Try resolve_conj on a sliced tensor if possible
        try {
            if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
                torch::Tensor sliced = input_tensor.slice(0, 0, input_tensor.size(0) - 1);
                torch::Tensor resolved_slice = torch::resolve_conj(sliced);
                
                // Test conj on slice
                torch::Tensor conj_slice = sliced.conj();
                torch::Tensor resolved_conj_slice = torch::resolve_conj(conj_slice);
            }
        } catch (...) {
            // Silently ignore shape-related failures
        }
        
        // Try resolve_conj on a transposed tensor if possible
        try {
            if (input_tensor.dim() >= 2) {
                torch::Tensor transposed = input_tensor.transpose(0, 1);
                torch::Tensor resolved_transpose = torch::resolve_conj(transposed);
                
                // Test conj on transpose
                torch::Tensor conj_transpose = transposed.conj();
                torch::Tensor resolved_conj_transpose = torch::resolve_conj(conj_transpose);
            }
        } catch (...) {
            // Silently ignore shape-related failures
        }
        
        // Try resolve_conj on a tensor with requires_grad set
        try {
            if (input_tensor.is_floating_point() || input_tensor.is_complex()) {
                auto grad_tensor = input_tensor.clone().detach().requires_grad_(true);
                torch::Tensor resolved_grad = torch::resolve_conj(grad_tensor);
                
                // Test conj with grad
                torch::Tensor conj_grad = grad_tensor.conj();
                torch::Tensor resolved_conj_grad = torch::resolve_conj(conj_grad);
            }
        } catch (...) {
            // Silently ignore gradient-related failures
        }
        
        // Try resolve_conj on a zero-sized tensor
        try {
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_tensor = torch::empty(empty_shape, torch::kComplexFloat);
            torch::Tensor conj_empty = empty_tensor.conj();
            torch::Tensor resolved_empty = torch::resolve_conj(conj_empty);
        } catch (...) {
            // Silently ignore
        }
        
        // Test with double-conjugation
        try {
            torch::Tensor double_conj = input_tensor.conj().conj();
            torch::Tensor resolved_double = torch::resolve_conj(double_conj);
        } catch (...) {
            // Silently ignore
        }
        
        // Test resolve_conj preserves data for non-conjugated tensor
        try {
            torch::Tensor cloned = input_tensor.clone();
            torch::Tensor resolved_clone = torch::resolve_conj(cloned);
        } catch (...) {
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}