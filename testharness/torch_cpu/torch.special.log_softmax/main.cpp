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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to apply log_softmax along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Handle dimension selection based on tensor dimensionality
        if (input.dim() == 0) {
            // For scalar tensor, dim must be 0 or -1
            dim = 0;
        } else {
            // Use modulo to ensure dim is valid (handle negative values properly)
            dim = dim % input.dim();
            if (dim < 0) {
                dim += input.dim();
            }
        }
        
        // Ensure input is floating point for log_softmax
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply log_softmax operation using functional API
        // torch::special::log_softmax may not exist, use torch::log_softmax instead
        torch::Tensor output;
        try {
            output = torch::log_softmax(input, dim, std::nullopt);
        } catch (...) {
            // Shape/dimension errors are expected for some inputs
        }
        
        // Try with optional dtype parameter if we have more data
        if (offset + 1 <= Size) {
            uint8_t dtype_selector = Data[offset++];
            
            // Only use floating point dtypes for log_softmax
            torch::ScalarType dtype;
            switch (dtype_selector % 4) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kFloat16; break;
                case 3: dtype = torch::kBFloat16; break;
                default: dtype = torch::kFloat32; break;
            }
            
            try {
                // Apply log_softmax with dtype
                torch::Tensor output_with_dtype = torch::log_softmax(input, dim, dtype);
            } catch (...) {
                // Expected for some dtype combinations
            }
        }
        
        // Test with negative dimension indexing
        if (input.dim() > 0) {
            try {
                torch::Tensor output_neg_dim = torch::log_softmax(input, -1, std::nullopt);
            } catch (...) {
                // Expected for some inputs
            }
        }
        
        // Test different dimensions if tensor has multiple dims
        if (input.dim() > 1) {
            for (int64_t d = 0; d < std::min(input.dim(), (int64_t)3); d++) {
                try {
                    torch::Tensor output_d = torch::log_softmax(input, d, std::nullopt);
                } catch (...) {
                    // Expected for some inputs
                }
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