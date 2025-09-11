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
        
        // Need at least a few bytes to create a tensor and source/destination axes
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get tensor rank (number of dimensions)
        int64_t rank = input_tensor.dim();
        
        // If we've consumed all data or not enough left for axes, return
        if (offset + 2 >= Size) {
            return 0;
        }
        
        // Parse source and destination axes
        int64_t source_axis = static_cast<int8_t>(Data[offset++]); // Use int8_t to allow negative values
        int64_t destination_axis = static_cast<int8_t>(Data[offset++]);
        
        // Test single axis moveaxis
        try {
            torch::Tensor result1 = torch::moveaxis(input_tensor, source_axis, destination_axis);
        } catch (const c10::Error &e) {
            // PyTorch-specific exceptions are expected for invalid inputs
        }
        
        // If we have enough data, test multiple axes moveaxis
        if (offset + 2 < Size && rank > 1) {
            std::vector<int64_t> source_axes;
            std::vector<int64_t> destination_axes;
            
            // Get number of axes to move (limited by rank)
            uint8_t num_axes = Data[offset++] % rank + 1;
            
            // Ensure we have enough data for the axes
            if (offset + 2 * num_axes <= Size) {
                for (uint8_t i = 0; i < num_axes; ++i) {
                    if (offset + 1 < Size) {
                        source_axes.push_back(static_cast<int8_t>(Data[offset++]));
                        destination_axes.push_back(static_cast<int8_t>(Data[offset++]));
                    }
                }
                
                // Test multiple axes moveaxis
                try {
                    torch::Tensor result2 = torch::moveaxis(input_tensor, source_axes, destination_axes);
                } catch (const c10::Error &e) {
                    // PyTorch-specific exceptions are expected for invalid inputs
                }
            }
        }
        
        // Test edge cases with empty tensors if we have a tensor with at least one dimension
        if (rank > 0) {
            // Create an empty tensor with the same dtype
            std::vector<int64_t> empty_shape = input_tensor.sizes().vec();
            empty_shape[0] = 0;
            torch::Tensor empty_tensor = torch::empty(empty_shape, input_tensor.options());
            
            try {
                torch::Tensor empty_result = torch::moveaxis(empty_tensor, source_axis, destination_axis);
            } catch (const c10::Error &e) {
                // PyTorch-specific exceptions are expected for invalid inputs
            }
        }
        
        // Test with scalar tensor (0-dim)
        if (offset < Size) {
            torch::Tensor scalar_tensor = torch::tensor(static_cast<int>(Data[offset]));
            try {
                torch::Tensor scalar_result = torch::moveaxis(scalar_tensor, source_axis, destination_axis);
            } catch (const c10::Error &e) {
                // PyTorch-specific exceptions are expected for invalid inputs
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
