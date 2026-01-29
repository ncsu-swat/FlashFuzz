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
        
        // Skip 0-dimensional tensors (can't unbind them)
        if (input_tensor.dim() == 0) {
            return 0;
        }
        
        // Get dimension to unbind along, bounded by actual tensor dimensions
        int64_t dim = 0;
        if (offset < Size) {
            // Use single byte and map to valid dimension range
            dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
            
            // Allow negative dimensions too (50% chance)
            if (offset < Size && (Data[offset++] % 2 == 0)) {
                dim = dim - input_tensor.dim();
            }
        }
        
        // Apply unbind operation
        std::vector<torch::Tensor> result;
        
        // Try different variants of unbind
        if (offset < Size) {
            uint8_t variant = Data[offset++];
            
            switch (variant % 2) {
                case 0:
                    // Basic unbind with dimension
                    result = torch::unbind(input_tensor, dim);
                    break;
                    
                case 1:
                    // Unbind with default dimension (0)
                    result = torch::unbind(input_tensor);
                    break;
            }
        } else {
            // Default case if we don't have enough data for variant
            result = torch::unbind(input_tensor, dim);
        }
        
        // Perform some operations on the result to ensure it's used
        if (!result.empty()) {
            for (auto& tensor : result) {
                auto sizes = tensor.sizes();
                auto numel = tensor.numel();
                auto dtype = tensor.dtype();
                
                // Simple operation to ensure tensor is valid
                if (numel > 0) {
                    try {
                        // Inner try-catch for expected operation failures
                        tensor = tensor + 1;
                    } catch (...) {
                        // Silently ignore operation failures on result tensors
                    }
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}