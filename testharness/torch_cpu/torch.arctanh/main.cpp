#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
            input_copy.arctanh_();
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
            
            // Convert input to different dtype and apply arctanh
            torch::Tensor input_cast = input.to(dtype);
            torch::Tensor result_cast = torch::arctanh(input_cast);
        }
        
        // Try with values at the boundaries of arctanh domain (-1, 1)
        if (offset < Size) {
            std::vector<double> boundary_values = {-0.9999, -0.5, 0.0, 0.5, 0.9999};
            for (double val : boundary_values) {
                torch::Tensor boundary_tensor = torch::full_like(input, val);
                torch::Tensor boundary_result = torch::arctanh(boundary_tensor);
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