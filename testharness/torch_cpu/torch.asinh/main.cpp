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
        
        // Apply asinh operation
        torch::Tensor result = torch::asinh(input);
        
        // Try in-place version if there's more data
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor input_copy = input.clone();
            input_copy.asinh_();
        }
        
        // Try with different output types if there's more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Apply asinh and convert to desired dtype
            torch::Tensor result_with_dtype = torch::asinh(input).to(dtype);
        }
        
        // Try with named tensor if there's more data
        if (offset < Size && Data[offset] % 3 == 0) {
            if (input.dim() > 0) {
                std::vector<torch::Dimname> names;
                for (int64_t i = 0; i < input.dim(); ++i) {
                    std::string name_str = "dim" + std::to_string(i);
                    names.push_back(torch::Dimname::fromSymbol(torch::Symbol::dimname(name_str)));
                }
                
                auto named_input = input.refine_names(names);
                auto named_result = torch::asinh(named_input);
            }
        }
        
        // Try with out parameter if there's more data
        if (offset < Size && Data[offset] % 5 == 0) {
            torch::Tensor out = torch::empty_like(input);
            torch::asinh_out(out, input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
