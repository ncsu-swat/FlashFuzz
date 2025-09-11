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
        
        // Parse optional parameters from the remaining data
        bool normalize = false;
        std::vector<int64_t> dim;
        
        // Parse normalize flag if we have more data
        if (offset < Size) {
            normalize = Data[offset++] & 0x1;
        }
        
        // Parse dimensions for ifft2
        if (offset < Size) {
            uint8_t num_dims = Data[offset++] % 3; // Up to 2 dimensions for ifft2
            
            for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                int64_t dimension = static_cast<int64_t>(Data[offset++]);
                dim.push_back(dimension);
            }
        }
        
        // If no dimensions specified, use default (-2, -1)
        if (dim.empty()) {
            dim = {-2, -1};
        }
        
        // Apply ifft2 operation
        torch::Tensor output;
        
        // Handle different parameter combinations
        if (dim.size() == 2) {
            output = torch::fft::ifft2(input_tensor, dim, normalize);
        } else if (dim.size() == 1) {
            // If only one dimension provided, use it and -1
            std::vector<int64_t> adjusted_dim = {dim[0], -1};
            output = torch::fft::ifft2(input_tensor, adjusted_dim, normalize);
        } else {
            // Default case
            output = torch::fft::ifft2(input_tensor, normalize);
        }
        
        // Perform some operation on the output to ensure it's used
        auto sum = output.sum();
        
        // Try to access elements if tensor is not empty
        if (output.numel() > 0) {
            auto item = output.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
