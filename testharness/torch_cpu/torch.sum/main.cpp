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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for sum operation if we have more data
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            // Extract dimension parameter
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Use the dimension as is - let PyTorch handle invalid dimensions
            
            // Extract keepdim parameter if we have more data
            if (offset < Size) {
                keepdim = Data[offset++] & 0x1; // Use lowest bit to determine boolean value
            }
        }
        
        // Try different variants of torch::sum
        try {
            // Variant 1: Sum over all dimensions
            torch::Tensor result1 = torch::sum(input);
            
            // Variant 2: Sum over specific dimension
            torch::Tensor result2 = torch::sum(input, dim);
            
            // Variant 3: Sum over specific dimension with keepdim
            torch::Tensor result3 = torch::sum(input, dim, keepdim);
            
            // Variant 4: Sum with dtype specified
            if (offset < Size) {
                auto dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                torch::Tensor result4 = torch::sum(input, torch::ScalarType(dtype));
            }
            
            // Variant 5: Sum with named dimension (if tensor has named dimensions)
            if (input.dim() > 0) {
                try {
                    // This might throw if named dimensions aren't supported
                    at::Dimname dimname = at::Dimname::fromSymbol(at::Symbol::dimname("dim_0"));
                    torch::Tensor result5 = torch::sum(input, {dimname});
                } catch (...) {
                    // Ignore errors for named dimensions
                }
            }
            
            // Variant 6: Sum with out tensor
            if (input.dim() > 0) {
                try {
                    auto out_options = torch::TensorOptions().dtype(input.dtype());
                    torch::Tensor out = torch::empty({}, out_options);
                    torch::sum_out(out, input);
                } catch (...) {
                    // Ignore errors for out tensor
                }
            }
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected and part of testing
            // We don't want to discard these inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
