#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
#include <cstring>        // For std::memcpy

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
        
        // Need at least a few bytes to create a tensor and specify dim
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip 0-dimensional tensors as cumsum requires at least 1 dimension
        if (input.dim() == 0) {
            return 0;
        }
        
        // Get a dimension to perform cumsum along
        int64_t dim = 0;
        if (offset < Size) {
            // Extract a dimension from the input data
            int64_t raw_dim = 0;
            size_t bytes_to_read = std::min(sizeof(int64_t), Size - offset);
            std::memcpy(&raw_dim, Data + offset, bytes_to_read);
            offset += bytes_to_read;
            
            // Use modulo to ensure dim is within valid range [-ndim, ndim-1]
            int64_t ndim = input.dim();
            dim = ((raw_dim % ndim) + ndim) % ndim;  // Normalize to [0, ndim-1]
            
            // Occasionally test negative dimension indexing
            if (offset < Size && Data[offset - 1] % 4 == 0) {
                dim = dim - ndim;  // Convert to negative [-ndim, -1]
            }
        }
        
        // Get dtype for output (optional)
        torch::ScalarType dtype = input.scalar_type();
        bool use_dtype = false;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
            use_dtype = true;
        }
        
        // Apply cumsum operation
        torch::Tensor output;
        
        // Determine which variant to test
        int variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 3;
        }
        
        try {
            if (variant == 0) {
                // Variant 1: cumsum with dimension only
                output = torch::cumsum(input, dim);
            } 
            else if (variant == 1 && use_dtype) {
                // Variant 2: cumsum with dimension and dtype
                output = torch::cumsum(input, dim, dtype);
            }
            else {
                // Variant 3: cumsum using Tensor method
                output = input.cumsum(dim);
            }
            
            // Perform a simple operation on the result to ensure it's used
            auto sum = output.sum();
            (void)sum;  // Suppress unused variable warning
        }
        catch (const c10::Error &e) {
            // Expected failures (e.g., dtype conversion issues) - silently ignore
        }
        
        // Test in-place version if we have more data
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                // Clone to a compatible dtype for in-place operation
                torch::Tensor input_copy = input.clone();
                // cumsum_ requires floating point or complex type
                if (input_copy.is_floating_point() || input_copy.is_complex()) {
                    input_copy.cumsum_(dim);
                }
            }
            catch (const c10::Error &e) {
                // Expected failures for in-place - silently ignore
            }
        }
        
        // Test with out tensor if we have more data
        if (offset + 1 < Size && Data[offset] % 3 == 0) {
            try {
                torch::Tensor out_tensor = torch::empty_like(input);
                torch::cumsum_out(out_tensor, input, dim);
                (void)out_tensor.sum();  // Use the result
            }
            catch (const c10::Error &e) {
                // Expected failures - silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}