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
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor with various data types
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply is_floating_point operation
        bool is_float = torch::is_floating_point(tensor);
        
        // Use the result to prevent optimization and verify behavior
        volatile bool result = is_float;
        (void)result;
        
        // Try with a view of the tensor if possible
        if (tensor.dim() > 0 && tensor.numel() > 0) {
            try {
                torch::Tensor view_tensor = tensor.view({-1});
                volatile bool is_view_float = torch::is_floating_point(view_tensor);
                (void)is_view_float;
            } catch (...) {
                // View may fail for some tensor configurations, silently ignore
            }
        }
        
        // Try with a slice of the tensor if possible
        if (tensor.dim() > 0 && tensor.size(0) > 1) {
            try {
                torch::Tensor slice_tensor = tensor.slice(0, 0, tensor.size(0) / 2 + 1);
                volatile bool is_slice_float = torch::is_floating_point(slice_tensor);
                (void)is_slice_float;
            } catch (...) {
                // Slice may fail for some configurations, silently ignore
            }
        }
        
        // Try with a transposed tensor if possible
        if (tensor.dim() >= 2) {
            try {
                torch::Tensor transposed = tensor.transpose(0, tensor.dim() - 1);
                volatile bool is_transposed_float = torch::is_floating_point(transposed);
                (void)is_transposed_float;
            } catch (...) {
                // Transpose may fail, silently ignore
            }
        }
        
        // Try with a contiguous tensor
        torch::Tensor contiguous = tensor.contiguous();
        volatile bool is_contiguous_float = torch::is_floating_point(contiguous);
        (void)is_contiguous_float;
        
        // Try with a clone
        torch::Tensor clone = tensor.clone();
        volatile bool is_clone_float = torch::is_floating_point(clone);
        (void)is_clone_float;
        
        // Try with a detached tensor
        torch::Tensor detached = tensor.detach();
        volatile bool is_detached_float = torch::is_floating_point(detached);
        (void)is_detached_float;
        
        // Test with explicitly created tensors of different dtypes to ensure coverage
        if (Size > 4) {
            uint8_t dtype_selector = Data[offset % Size];
            
            // Test with integer types
            if (dtype_selector % 4 == 0) {
                torch::Tensor int_tensor = torch::zeros({2, 2}, torch::kInt32);
                volatile bool int_result = torch::is_floating_point(int_tensor);
                (void)int_result;
            }
            // Test with float types
            else if (dtype_selector % 4 == 1) {
                torch::Tensor float_tensor = torch::zeros({2, 2}, torch::kFloat32);
                volatile bool float_result = torch::is_floating_point(float_tensor);
                (void)float_result;
            }
            // Test with double
            else if (dtype_selector % 4 == 2) {
                torch::Tensor double_tensor = torch::zeros({2, 2}, torch::kFloat64);
                volatile bool double_result = torch::is_floating_point(double_tensor);
                (void)double_result;
            }
            // Test with bool
            else {
                torch::Tensor bool_tensor = torch::zeros({2, 2}, torch::kBool);
                volatile bool bool_result = torch::is_floating_point(bool_tensor);
                (void)bool_result;
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