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
        
        // Need at least some data to create a tensor
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.trace operation
        // trace() requires a 2D tensor, but we'll let PyTorch handle the error cases
        torch::Tensor result;
        
        // Try to apply trace operation
        result = torch::trace(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined()) {
            float value = result.item<float>();
            (void)value; // Prevent unused variable warning
        }
        
        // Try with different variants if we have more data
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            // Try with different tensor shapes if possible
            if (variant % 3 == 0 && offset < Size) {
                // Create a square matrix if possible
                int64_t dim = (Data[offset++] % 10) + 1;
                torch::Tensor square_matrix = torch::ones({dim, dim});
                torch::Tensor trace_result = torch::trace(square_matrix);
            }
            
            // Try with non-square matrix
            if (variant % 3 == 1 && offset < Size) {
                int64_t rows = (Data[offset++] % 10) + 1;
                int64_t cols = rows;
                if (offset < Size) {
                    cols = (Data[offset++] % 10) + 1;
                }
                torch::Tensor non_square = torch::ones({rows, cols});
                torch::Tensor trace_result = torch::trace(non_square);
            }
            
            // Try with different data types
            if (variant % 3 == 2 && offset < Size) {
                uint8_t dtype_selector = Data[offset++];
                torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                // Create a tensor with the selected data type
                torch::Tensor typed_tensor = torch::ones({2, 2}, torch::TensorOptions().dtype(dtype));
                
                // Apply trace
                torch::Tensor trace_result = torch::trace(typed_tensor);
            }
        }
        
        // Try with edge cases
        // Empty tensor (0x0)
        torch::Tensor empty_matrix = torch::empty({0, 0});
        try {
            torch::Tensor trace_empty = torch::trace(empty_matrix);
        } catch (...) {
            // Ignore exceptions for empty tensor
        }
        
        // 1D tensor (should fail, but let PyTorch handle it)
        if (input_tensor.dim() == 1) {
            try {
                torch::Tensor trace_1d = torch::trace(input_tensor);
            } catch (...) {
                // Ignore exceptions for 1D tensor
            }
        }
        
        // 3D+ tensor (should fail, but let PyTorch handle it)
        if (input_tensor.dim() > 2) {
            try {
                torch::Tensor trace_3d = torch::trace(input_tensor);
            } catch (...) {
                // Ignore exceptions for higher-dimensional tensors
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
