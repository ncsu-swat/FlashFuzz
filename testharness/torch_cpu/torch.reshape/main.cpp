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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract reshape parameters from the remaining data
        std::vector<int64_t> new_shape;
        
        // Determine number of dimensions for reshape
        if (offset < Size) {
            uint8_t num_dims = Data[offset++] % 6; // 0-5 dimensions
            
            // Parse each dimension
            for (uint8_t i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; ++i) {
                int64_t dim_value;
                std::memcpy(&dim_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Allow negative dimensions for reshape (which have special meaning)
                new_shape.push_back(dim_value);
            }
            
            // If we have at least one dimension left, use it for the special -1 value
            if (offset < Size && new_shape.size() > 0) {
                uint8_t use_special = Data[offset++] % 2;
                if (use_special) {
                    // Replace a random dimension with -1 (auto-infer)
                    size_t idx = Data[offset % new_shape.size()];
                    new_shape[idx] = -1;
                }
            }
        }
        
        // If we couldn't extract any shape, use some defaults
        if (new_shape.empty() && input_tensor.numel() > 0) {
            // Try some common reshape patterns
            if (offset < Size) {
                uint8_t pattern = Data[offset++] % 4;
                switch (pattern) {
                    case 0: // Flatten
                        new_shape.push_back(-1);
                        break;
                    case 1: // Make 2D
                        new_shape.push_back(-1);
                        new_shape.push_back(1);
                        break;
                    case 2: // Make 3D
                        new_shape.push_back(-1);
                        new_shape.push_back(1);
                        new_shape.push_back(1);
                        break;
                    case 3: // Empty tensor
                        if (input_tensor.numel() == 0) {
                            new_shape.push_back(0);
                        } else {
                            new_shape.push_back(-1);
                        }
                        break;
                }
            } else {
                // Default to flatten
                new_shape.push_back(-1);
            }
        }
        
        // Apply reshape operation
        torch::Tensor output;
        
        // Try different reshape variants
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 3;
            
            switch (variant) {
                case 0:
                    // Standard reshape
                    output = input_tensor.reshape(new_shape);
                    break;
                case 1:
                    // View (when possible)
                    try {
                        output = input_tensor.view(new_shape);
                    } catch (const std::exception&) {
                        // Fallback to reshape if view fails
                        output = input_tensor.reshape(new_shape);
                    }
                    break;
                case 2:
                    // Functional API
                    output = torch::reshape(input_tensor, new_shape);
                    break;
            }
        } else {
            // Default to standard reshape
            output = input_tensor.reshape(new_shape);
        }
        
        // Basic validation - number of elements should match
        if (input_tensor.numel() != output.numel()) {
            throw std::runtime_error("Element count mismatch after reshape");
        }
        
        // Force evaluation
        output.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
