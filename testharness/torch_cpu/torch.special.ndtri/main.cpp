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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input values are in valid range for ndtri (0 to 1)
        // ndtri is the inverse of the normal CDF, so inputs must be probabilities
        torch::Tensor clamped_input = torch::clamp(input, 0.0001, 0.9999);
        
        // Apply the torch.special.ndtri operation
        torch::Tensor result = torch::special::ndtri(clamped_input);
        
        // Try with unclamped input to test edge cases (may produce inf/-inf for 0/1)
        if (offset < Size) {
            try {
                torch::Tensor edge_result = torch::special::ndtri(input);
                (void)edge_result;
            } catch (const std::exception &e) {
                // Expected exceptions for invalid inputs are fine
            }
        }
        
        // Try with scalar inputs if we have more data
        if (offset + 1 < Size) {
            double scalar_val = static_cast<double>(Data[offset]) / 255.0;
            offset++;
            try {
                torch::Tensor scalar_tensor = torch::tensor(scalar_val);
                torch::Tensor scalar_result = torch::special::ndtri(scalar_tensor);
                (void)scalar_result;
            } catch (const std::exception &e) {
                // Expected exceptions for invalid inputs are fine
            }
        }
        
        // Try with different tensor types if we have more data
        if (offset < Size) {
            try {
                torch::Tensor float_input = clamped_input.to(torch::kFloat32);
                torch::Tensor float_result = torch::special::ndtri(float_input);
                (void)float_result;
                
                torch::Tensor double_input = clamped_input.to(torch::kFloat64);
                torch::Tensor double_result = torch::special::ndtri(double_input);
                (void)double_result;
            } catch (const std::exception &e) {
                // Expected exceptions for unsupported types are fine
            }
            
            // Try half precision separately as it may not be supported
            try {
                torch::Tensor half_input = clamped_input.to(torch::kFloat16);
                torch::Tensor half_result = torch::special::ndtri(half_input);
                (void)half_result;
            } catch (const std::exception &e) {
                // Half precision may not be supported
            }
        }
        
        // Try with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0}, torch::kFloat32);
            torch::Tensor empty_result = torch::special::ndtri(empty_tensor);
            (void)empty_result;
        } catch (const std::exception &e) {
            // Expected exceptions for empty tensors are fine
        }
        
        // Try with multi-dimensional tensor
        if (offset + 4 <= Size) {
            try {
                int64_t dim0 = (Data[offset] % 4) + 1;
                int64_t dim1 = (Data[offset + 1] % 4) + 1;
                torch::Tensor multi_dim = torch::rand({dim0, dim1});
                torch::Tensor multi_result = torch::special::ndtri(multi_dim);
                (void)multi_result;
            } catch (const std::exception &e) {
                // Handle any exceptions
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