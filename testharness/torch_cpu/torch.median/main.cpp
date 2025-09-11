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
        
        // Get a dimension value for median if tensor is not a scalar
        int64_t dim = 0;
        bool keepdim = false;
        
        // If we have more data, use it to determine dimension and keepdim
        if (offset + 2 <= Size) {
            dim = static_cast<int64_t>(Data[offset++]) % (input.dim() + 1) - 1; // -1 means no dim specified
            keepdim = Data[offset++] & 0x1; // Use lowest bit for boolean
        }
        
        // Try different variants of median
        try {
            // Variant 1: median without dimension (returns single value)
            torch::Tensor result1 = torch::median(input);
            
            // Variant 2: median with dimension (returns tuple)
            if (input.dim() > 0 && dim >= 0) {
                auto result2 = torch::median(input, dim, keepdim);
                auto values = std::get<0>(result2);
                auto indices = std::get<1>(result2);
            }
            
            // Variant 3: median with named dimension (if available)
            if (input.dim() > 0 && offset < Size) {
                // Try to create a named tensor if we have more data
                std::vector<torch::Dimname> names;
                for (int64_t i = 0; i < input.dim() && offset < Size; i++) {
                    char name_char = static_cast<char>('a' + (Data[offset++] % 26));
                    names.push_back(torch::Dimname::fromSymbol(torch::Symbol::dimname(std::string(1, name_char))));
                }
                
                if (names.size() == input.dim()) {
                    torch::Tensor named_input = input.refine_names(names);
                    if (dim >= 0 && dim < named_input.dim()) {
                        auto result3 = torch::median(named_input, names[dim], keepdim);
                        auto values = std::get<0>(result3);
                        auto indices = std::get<1>(result3);
                    }
                }
            }
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected and okay
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
