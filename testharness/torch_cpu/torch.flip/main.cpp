#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse dimensions to flip
        std::vector<int64_t> dims_to_flip;
        
        // Determine number of dimensions to flip
        if (offset < Size) {
            uint8_t num_dims_to_flip = Data[offset++] % (input_tensor.dim() + 1);
            
            // Parse each dimension to flip
            for (uint8_t i = 0; i < num_dims_to_flip && offset < Size; ++i) {
                // Allow negative dimensions to test edge cases
                int64_t dim;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&dim, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                } else {
                    // Not enough data, use a simple byte
                    dim = static_cast<int64_t>(Data[offset++]);
                }
                
                dims_to_flip.push_back(dim);
            }
        }
        
        // If no dimensions were specified, use a default
        if (dims_to_flip.empty() && input_tensor.dim() > 0) {
            dims_to_flip.push_back(0);
        }
        
        // Apply torch.flip operation
        torch::Tensor output;
        
        // Test with different dimension configurations
        if (input_tensor.dim() == 0) {
            // Scalar tensor - flip should throw an exception or return the same tensor
            output = torch::flip(input_tensor, {});
        } else {
            output = torch::flip(input_tensor, dims_to_flip);
        }
        
        // Test some properties of the output tensor
        if (output.sizes() != input_tensor.sizes()) {
            throw std::runtime_error("Output tensor has different shape than input tensor");
        }
        
        // Try accessing elements to ensure the tensor is valid
        if (output.numel() > 0) {
            output.item();
        }
        
        // Try additional operations on the flipped tensor
        if (output.dim() > 0 && output.numel() > 0) {
            // Try another flip
            torch::Tensor double_flip = torch::flip(output, dims_to_flip);
            
            // Try some basic operations
            torch::Tensor sum = output.sum();
            torch::Tensor mean = output.mean();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}