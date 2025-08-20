#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for pdist
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // pdist requires a 2D tensor with shape [N, M] where N >= 1 and M >= 1
        // If the tensor doesn't have the right shape, reshape it
        if (input.dim() != 2) {
            // If tensor is empty, create a small valid tensor
            if (input.numel() == 0) {
                input = torch::ones({2, 2});
            } else {
                // Reshape to 2D tensor
                int64_t numel = input.numel();
                int64_t dim1 = std::max(static_cast<int64_t>(1), static_cast<int64_t>(std::sqrt(numel)));
                int64_t dim2 = (numel + dim1 - 1) / dim1; // Ceiling division
                input = input.reshape({dim1, dim2});
            }
        }
        
        // Get a p-norm value from the input data
        double p = 2.0; // Default to Euclidean distance
        if (offset < Size) {
            // Use the next byte to determine p
            uint8_t p_byte = Data[offset++];
            
            // Map to common p-norm values or use raw value
            switch (p_byte % 5) {
                case 0: p = 0.0; break;  // Test with p=0
                case 1: p = 1.0; break;  // Manhattan distance
                case 2: p = 2.0; break;  // Euclidean distance
                case 3: p = std::numeric_limits<double>::infinity(); break; // Infinity norm
                case 4: 
                    // Use a value from the data
                    if (offset < Size) {
                        p = static_cast<double>(Data[offset++]) / 10.0;
                    }
                    break;
            }
        }
        
        // Apply pdist operation
        torch::Tensor output = torch::pdist(input, p);
        
        // Try with different p values if we have more data
        if (offset + 1 < Size) {
            double p2 = static_cast<double>(Data[offset]) / 10.0;
            torch::Tensor output2 = torch::pdist(input, p2);
        }
        
        // Test edge cases with specific shapes if we have enough data
        if (offset + 2 < Size) {
            uint8_t shape_selector = Data[offset++];
            
            // Create tensors with specific shapes for edge cases
            torch::Tensor edge_input;
            
            switch (shape_selector % 5) {
                case 0:
                    // Single row (should result in empty output)
                    edge_input = torch::ones({1, 2});
                    break;
                case 1:
                    // Two identical rows (should result in zero distances)
                    edge_input = torch::ones({2, 3});
                    break;
                case 2:
                    // Large number of rows
                    edge_input = torch::ones({100, 2});
                    break;
                case 3:
                    // High dimensional features
                    edge_input = torch::ones({5, 50});
                    break;
                case 4:
                    // Minimal valid case
                    edge_input = torch::ones({2, 1});
                    break;
            }
            
            // Apply pdist to the edge case
            torch::Tensor edge_output = torch::pdist(edge_input, p);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}