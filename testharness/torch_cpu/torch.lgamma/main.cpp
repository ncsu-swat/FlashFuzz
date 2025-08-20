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
        
        // Apply lgamma operation
        torch::Tensor result = torch::lgamma(input);
        
        // Try in-place version if there's enough data to decide
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            input_copy.lgamma_();
        }
        
        // Try with named tensor if there's enough data
        if (offset + 1 < Size && input.dim() > 0) {
            uint8_t name_selector = Data[offset++] % 4; // Use modulo to limit options
            
            std::vector<torch::Dimname> names;
            for (int64_t i = 0; i < input.dim(); i++) {
                switch ((name_selector + i) % 4) {
                    case 0: names.push_back(torch::Dimname::fromSymbol(torch::Symbol::dimname("batch"))); break;
                    case 1: names.push_back(torch::Dimname::fromSymbol(torch::Symbol::dimname("channel"))); break;
                    case 2: names.push_back(torch::Dimname::fromSymbol(torch::Symbol::dimname("height"))); break;
                    case 3: names.push_back(torch::Dimname::fromSymbol(torch::Symbol::dimname("width"))); break;
                }
            }
            
            // Apply named tensor operations if possible
            try {
                torch::Tensor named_input = input.refine_names(names);
                torch::Tensor named_result = torch::lgamma(named_input);
            } catch (const std::exception&) {
                // Named tensor operations might fail for various reasons
                // Just continue with the fuzzing
            }
        }
        
        // Try with out parameter if there's enough data
        if (offset < Size) {
            torch::Tensor out = torch::empty_like(input);
            torch::lgamma_out(out, input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}