#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply bitwise_not operation
        torch::Tensor result = torch::bitwise_not(input_tensor);
        
        // Try inplace version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.bitwise_not_();
        }
        
        // Try with different tensor options if there's more data
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Create a tensor with different dtype
            torch::ScalarType dtype = fuzzer_utils::parseDataType(option_byte);
            
            // Try to convert the input tensor to the new dtype if possible
            try {
                torch::Tensor converted = input_tensor.to(dtype);
                torch::Tensor result2 = torch::bitwise_not(converted);
            } catch (const std::exception&) {
                // Some dtype conversions may not be valid for bitwise operations
            }
        }
        
        // Try with named tensor if there's more data
        if (offset + 1 < Size) {
            try {
                torch::Tensor named_tensor = input_tensor.clone();
                std::vector<torch::Dimname> names;
                
                for (int64_t i = 0; i < named_tensor.dim(); i++) {
                    std::string dim_name = "dim" + std::to_string(i);
                    names.push_back(torch::Dimname::fromSymbol(c10::Symbol::dimname(dim_name)));
                }
                
                if (!names.empty()) {
                    named_tensor = named_tensor.refine_names(names);
                    torch::Tensor named_result = torch::bitwise_not(named_tensor);
                }
            } catch (const std::exception&) {
                // Named tensors might not work with all shapes/dtypes
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