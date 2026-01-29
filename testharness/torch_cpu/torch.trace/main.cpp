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
        
        // Need at least some data to create a tensor
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch::trace requires a 2D tensor
        // Try to apply trace operation - let PyTorch validate dimensions
        try {
            torch::Tensor result = torch::trace(input_tensor);
            
            // Access the result to ensure computation is performed
            if (result.defined()) {
                // Use toDouble() which handles more dtypes safely
                volatile double value = result.item<double>();
                (void)value;
            }
        } catch (...) {
            // Silently ignore dimension/shape errors for random input tensors
        }
        
        // Try with different variants if we have more data
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            // Try with different tensor shapes if possible
            if (variant % 4 == 0 && offset < Size) {
                // Create a square matrix
                int64_t dim = (Data[offset++] % 10) + 1;
                torch::Tensor square_matrix = torch::randn({dim, dim});
                torch::Tensor trace_result = torch::trace(square_matrix);
                volatile double v = trace_result.item<double>();
                (void)v;
            }
            
            // Try with non-square matrix (trace still works, uses min(rows, cols))
            if (variant % 4 == 1 && offset + 1 < Size) {
                int64_t rows = (Data[offset++] % 10) + 1;
                int64_t cols = (Data[offset++] % 10) + 1;
                torch::Tensor non_square = torch::randn({rows, cols});
                torch::Tensor trace_result = torch::trace(non_square);
                volatile double v = trace_result.item<double>();
                (void)v;
            }
            
            // Try with different data types
            if (variant % 4 == 2 && offset < Size) {
                uint8_t dtype_selector = Data[offset++];
                torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                // Skip complex types for simpler handling
                if (dtype != torch::kComplexFloat && dtype != torch::kComplexDouble) {
                    int64_t dim = (offset < Size) ? (Data[offset++] % 8) + 2 : 3;
                    torch::Tensor typed_tensor = torch::ones({dim, dim}, torch::TensorOptions().dtype(dtype));
                    torch::Tensor trace_result = torch::trace(typed_tensor);
                    (void)trace_result;
                }
            }
            
            // Try with larger matrices to exercise more code paths
            if (variant % 4 == 3 && offset + 1 < Size) {
                int64_t dim1 = (Data[offset++] % 50) + 10;
                int64_t dim2 = (Data[offset++] % 50) + 10;
                torch::Tensor large_matrix = torch::randn({dim1, dim2});
                torch::Tensor trace_result = torch::trace(large_matrix);
                volatile double v = trace_result.item<double>();
                (void)v;
            }
        }
        
        // Test with identity matrix (trace should equal dimension)
        if (offset < Size) {
            int64_t eye_dim = (Data[offset++] % 10) + 1;
            torch::Tensor eye_matrix = torch::eye(eye_dim);
            torch::Tensor trace_eye = torch::trace(eye_matrix);
            volatile double v = trace_eye.item<double>();
            (void)v;
        }
        
        // Test with diagonal matrix
        if (offset < Size) {
            int64_t diag_size = (Data[offset++] % 10) + 1;
            torch::Tensor diag_values = torch::randn({diag_size});
            torch::Tensor diag_matrix = torch::diag(diag_values);
            torch::Tensor trace_diag = torch::trace(diag_matrix);
            volatile double v = trace_diag.item<double>();
            (void)v;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}