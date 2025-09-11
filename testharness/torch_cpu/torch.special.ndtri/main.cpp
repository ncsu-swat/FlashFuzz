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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input values are in valid range for ndtri (0 to 1)
        // ndtri is the inverse of the normal CDF, so inputs must be probabilities
        torch::Tensor clamped_input = torch::clamp(input, 0.0001, 0.9999);
        
        // Apply the torch.special.ndtri operation
        torch::Tensor result = torch::special::ndtri(clamped_input);
        
        // Try with unclamped input to test edge cases
        if (offset < Size) {
            try {
                torch::Tensor edge_result = torch::special::ndtri(input);
            } catch (const std::exception &e) {
                // Expected exceptions for invalid inputs are fine
            }
        }
        
        // Try with scalar inputs if we have more data
        if (offset + 1 < Size) {
            double scalar_val = static_cast<double>(Data[offset]) / 255.0;
            try {
                torch::Tensor scalar_tensor = torch::tensor(scalar_val);
                torch::Tensor scalar_result = torch::special::ndtri(scalar_tensor);
            } catch (const std::exception &e) {
                // Expected exceptions for invalid inputs are fine
            }
        }
        
        // Try with different tensor types if we have more data
        if (offset + 2 < Size) {
            try {
                torch::Tensor float_input = input.to(torch::kFloat);
                torch::Tensor float_result = torch::special::ndtri(float_input);
                
                torch::Tensor double_input = input.to(torch::kDouble);
                torch::Tensor double_result = torch::special::ndtri(double_input);
                
                // Try half precision if available
                torch::Tensor half_input = input.to(torch::kHalf);
                torch::Tensor half_result = torch::special::ndtri(half_input);
            } catch (const std::exception &e) {
                // Expected exceptions for invalid inputs are fine
            }
        }
        
        // Try with empty tensor
        if (offset + 1 < Size) {
            try {
                std::vector<int64_t> empty_shape = {0};
                torch::Tensor empty_tensor = torch::empty(empty_shape, torch::kFloat);
                torch::Tensor empty_result = torch::special::ndtri(empty_tensor);
            } catch (const std::exception &e) {
                // Expected exceptions for invalid inputs are fine
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
