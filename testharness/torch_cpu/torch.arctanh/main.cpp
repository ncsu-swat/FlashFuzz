#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
#include <cstdint>        // For uint64_t

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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply arctanh operation
        torch::Tensor result = torch::arctanh(input);
        
        // Try in-place version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            // Use the function form for in-place operation
            torch::arctanh_(input_copy);
        }
        
        // Try with out parameter if there's more data
        if (offset < Size) {
            torch::Tensor out = torch::empty_like(input);
            torch::arctanh_out(out, input);
        }
        
        // Try with different input types if there's more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Only apply to floating point types (arctanh requires float/double/complex)
            try {
                torch::Tensor input_cast = input.to(dtype);
                torch::Tensor result_cast = torch::arctanh(input_cast);
            } catch (...) {
                // Silently ignore dtype conversion failures (e.g., bool/int dtypes)
            }
        }
        
        // Try with values at the boundaries of arctanh domain (-1, 1)
        if (offset < Size) {
            std::vector<double> boundary_values = {-0.9999, -0.5, 0.0, 0.5, 0.9999};
            for (double val : boundary_values) {
                torch::Tensor boundary_tensor = torch::full_like(input, val);
                torch::Tensor boundary_result = torch::arctanh(boundary_tensor);
            }
        }
        
        // Test with special values that should produce inf/-inf
        if (offset < Size) {
            try {
                torch::Tensor edge_case = torch::tensor({-1.0, 1.0}, torch::kFloat32);
                torch::Tensor edge_result = torch::arctanh(edge_case);
            } catch (...) {
                // Silently ignore - edge cases may throw
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