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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply arcsin operation
        torch::Tensor result = torch::arcsin(input);
        
        // Try in-place version if there's enough data to decide
        if (offset < Size) {
            bool use_inplace = Data[offset++] % 2 == 0;
            if (use_inplace) {
                torch::Tensor input_copy = input.clone();
                input_copy.arcsin_();
            }
        }
        
        // Try with different options if there's more data
        if (offset + 1 < Size) {
            // Use the next byte to determine if we should test with out parameter
            bool use_out = Data[offset++] % 2 == 0;
            if (use_out) {
                // Create an output tensor with same shape and dtype
                torch::Tensor out = torch::empty_like(input);
                torch::arcsin_out(out, input);
            }
        }
        
        // Try with named tensor if there's more data
        if (offset < Size) {
            bool use_named = Data[offset++] % 2 == 0;
            if (use_named && input.dim() > 0) {
                // Create a named tensor
                std::vector<torch::Dimname> names;
                for (int64_t i = 0; i < input.dim(); ++i) {
                    std::string dim_name = "dim" + std::to_string(i);
                    names.push_back(torch::Dimname::fromSymbol(c10::Symbol::dimname(dim_name)));
                }
                
                torch::Tensor named_input = input.refine_names(names);
                torch::Tensor named_result = torch::arcsin(named_input);
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