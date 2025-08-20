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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a dimension value from the remaining data if available
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract keepdim boolean if data available
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x1);
            offset++;
        }
        
        // Test different variants of nanmedian
        
        // Variant 1: Basic nanmedian (no arguments)
        torch::Tensor result1 = torch::nanmedian(input);
        
        // Variant 2: nanmedian with dimension
        if (input.dim() > 0) {
            // Ensure dim is within valid range for the tensor
            dim = dim % std::max<int64_t>(1, input.dim());
            
            // Handle negative dim values
            if (dim < 0) {
                dim += input.dim();
            }
            
            // Call nanmedian with dimension
            auto result2 = torch::nanmedian(input, dim, keepdim);
            
            // Unpack the values and indices
            torch::Tensor values = std::get<0>(result2);
            torch::Tensor indices = std::get<1>(result2);
        }
        
        // Variant 3: nanmedian with named dimension
        if (input.dim() > 0 && !input.names().empty()) {
            try {
                // Get the first named dimension if available
                auto dimname = input.names()[0];
                auto result3 = torch::nanmedian(input, dimname, keepdim);
                
                // Unpack the values and indices
                torch::Tensor values = std::get<0>(result3);
                torch::Tensor indices = std::get<1>(result3);
            } catch (...) {
                // Ignore errors with named dimensions
            }
        }
        
        // Variant 4: Test with out parameter
        if (input.dim() > 0) {
            dim = dim % std::max<int64_t>(1, input.dim());
            if (dim < 0) {
                dim += input.dim();
            }
            
            // Create output tensors
            std::vector<int64_t> out_shape;
            if (keepdim) {
                out_shape = input.sizes().vec();
                out_shape[dim] = 1;
            } else {
                out_shape = input.sizes().vec();
                out_shape.erase(out_shape.begin() + dim);
            }
            
            torch::Tensor values_out = torch::empty(out_shape, input.options());
            torch::Tensor indices_out = torch::empty(out_shape, torch::kLong);
            
            // Call nanmedian with out parameter
            torch::nanmedian_out(values_out, indices_out, input, dim, keepdim);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}