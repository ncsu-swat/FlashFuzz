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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // flipud requires at least 1 dimension
        if (input_tensor.dim() < 1) {
            return 0;
        }
        
        // Apply flipud operation
        torch::Tensor result = torch::flipud(input_tensor);
        
        // Force computation by accessing sum instead of item()
        if (result.numel() > 0) {
            volatile auto sum = result.sum().item<float>();
            (void)sum;
        }
        
        // Try with a view of the tensor to test different memory layouts
        try {
            if (input_tensor.dim() > 1 && input_tensor.numel() > 0) {
                torch::Tensor view_tensor = input_tensor.transpose(0, input_tensor.dim() - 1);
                torch::Tensor view_result = torch::flipud(view_tensor);
                
                if (view_result.numel() > 0) {
                    volatile auto sum = view_result.sum().item<float>();
                    (void)sum;
                }
            }
        } catch (...) {
            // Silently ignore expected failures for edge cases
        }
        
        // Try with non-contiguous tensor
        try {
            if (input_tensor.dim() > 1 && input_tensor.size(0) > 1 && input_tensor.numel() > 0) {
                torch::Tensor non_contig = input_tensor.slice(0, 0, input_tensor.size(0), 2);
                torch::Tensor non_contig_result = torch::flipud(non_contig);
                
                if (non_contig_result.numel() > 0) {
                    volatile auto sum = non_contig_result.sum().item<float>();
                    (void)sum;
                }
            }
        } catch (...) {
            // Silently ignore expected failures for edge cases
        }
        
        // Try with 1D tensor explicitly
        try {
            if (offset + 4 < Size) {
                uint8_t dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                int64_t len = (Data[offset] % 16) + 1;
                offset++;
                torch::Tensor tensor_1d = torch::randn({len}, torch::TensorOptions().dtype(torch::kFloat32));
                torch::Tensor result_1d = torch::flipud(tensor_1d);
                volatile auto sum = result_1d.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Silently ignore
        }
        
        // Try with 2D tensor explicitly
        try {
            if (offset + 4 < Size) {
                int64_t rows = (Data[offset] % 8) + 1;
                offset++;
                int64_t cols = (Data[offset] % 8) + 1;
                offset++;
                torch::Tensor tensor_2d = torch::randn({rows, cols});
                torch::Tensor result_2d = torch::flipud(tensor_2d);
                volatile auto sum = result_2d.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Silently ignore
        }
        
        // Try with 3D tensor
        try {
            if (offset + 4 < Size) {
                int64_t d0 = (Data[offset] % 4) + 1;
                offset++;
                int64_t d1 = (Data[offset] % 4) + 1;
                offset++;
                int64_t d2 = (Data[offset] % 4) + 1;
                offset++;
                torch::Tensor tensor_3d = torch::randn({d0, d1, d2});
                torch::Tensor result_3d = torch::flipud(tensor_3d);
                volatile auto sum = result_3d.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Silently ignore
        }
        
        // Try different dtypes
        try {
            if (offset + 2 < Size) {
                torch::Tensor int_tensor = torch::randint(0, 100, {4, 4}, torch::kInt32);
                torch::Tensor int_result = torch::flipud(int_tensor);
                volatile auto sum = int_result.sum().item<int>();
                (void)sum;
            }
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