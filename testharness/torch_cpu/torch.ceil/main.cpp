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
        
        // Apply ceil operation
        torch::Tensor result = torch::ceil(input);
        
        // Try inplace version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            input_copy.ceil_();
        }
        
        // Try with out parameter if there's more data
        if (offset < Size) {
            torch::Tensor out = torch::empty_like(input);
            torch::ceil_out(out, input);
        }
        
        // Try with non-contiguous tensor if possible
        if (input.dim() > 1 && input.size(0) > 1) {
            torch::Tensor non_contiguous = input.transpose(0, input.dim() - 1);
            if (!non_contiguous.is_contiguous()) {
                torch::Tensor result_non_contiguous = torch::ceil(non_contiguous);
            }
        }
        
        // Try with different dtypes if there's more data
        if (offset < Size && Size - offset > 0) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Only try conversion if the dtype is different and is a valid type for ceil
            if (dtype != input.dtype() && 
                (dtype == torch::kFloat || dtype == torch::kDouble || 
                 dtype == torch::kHalf || dtype == torch::kBFloat16 ||
                 dtype == torch::kInt8 || dtype == torch::kInt16 || 
                 dtype == torch::kInt32 || dtype == torch::kInt64)) {
                try {
                    torch::Tensor converted = input.to(dtype);
                    torch::Tensor result_converted = torch::ceil(converted);
                } catch (const std::exception&) {
                    // Conversion might fail, that's okay
                }
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
