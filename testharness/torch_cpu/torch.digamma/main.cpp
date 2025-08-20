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
        
        // Create input tensor for digamma operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply digamma operation
        torch::Tensor result = torch::digamma(input);
        
        // Try in-place version if possible
        if (input.is_floating_point()) {
            // Create a copy for in-place operation
            torch::Tensor input_copy = input.clone();
            input_copy.digamma_();
        }
        
        // Try with different output tensor
        if (offset + 2 <= Size) {
            torch::Tensor output = fuzzer_utils::createTensor(Data, Size, offset);
            // Only proceed if output has same shape as input
            if (output.sizes() == input.sizes()) {
                torch::digamma_out(output, input);
            }
        }
        
        // Try with named tensor if we have enough data
        if (offset + 1 < Size) {
            // Create a named tensor version if possible
            if (input.dim() > 0) {
                std::vector<torch::Dimname> names;
                for (int64_t i = 0; i < input.dim(); ++i) {
                    names.push_back(torch::Dimname::fromSymbol(c10::Symbol::dimname("dim" + std::to_string(i))));
                }
                auto named_input = input.refine_names(names);
                auto named_result = torch::digamma(named_input);
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