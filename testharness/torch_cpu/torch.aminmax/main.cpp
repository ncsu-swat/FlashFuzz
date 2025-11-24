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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to use for aminmax if needed
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, ensure dim is within valid range
            if (input.dim() > 0) {
                dim = dim % input.dim();
                if (dim < 0) {
                    dim += input.dim();
                }
            }
        }
        
        // Get keepdim flag
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Test different variants of aminmax
        
        // Variant 1: aminmax without dimension (returns min/max over all elements)
        auto result1 = torch::aminmax(input);
        
        // Variant 2: aminmax with dimension
        if (input.dim() > 0) {
            auto result2 = torch::aminmax(input, dim, keepdim);
        }
        
        // Variant 3: aminmax with named dimension (if tensor has names)
        if (offset < Size && (Data[offset] % 2 == 0)) {
            // Try to set dimension names
            try {
                if (input.dim() > 0) {
                    std::vector<torch::Dimname> names;
                    for (int64_t i = 0; i < input.dim(); i++) {
                        names.push_back(torch::Dimname::wildcard());
                    }
                    auto named_tensor = input.refine_names(names);
                    auto result3 = torch::aminmax(named_tensor, 0, keepdim);
                }
            } catch (const std::exception&) {
                // Ignore exceptions from naming dimensions
            }
        }
        
        // Variant 4: out variant
        try {
            torch::Tensor min_out = torch::empty_like(input);
            torch::Tensor max_out = torch::empty_like(input);
            
            if (input.dim() > 0) {
                torch::aminmax_out(min_out, max_out, input, dim, keepdim);
            } else {
                // For scalars, we need different output shapes
                min_out = torch::empty({}, input.options());
                max_out = torch::empty({}, input.options());
                torch::aminmax_out(min_out, max_out, input);
            }
        } catch (const std::exception&) {
            // Ignore exceptions from out variant
        }
        
        // Variant 5: Test with empty tensor
        if (offset < Size && (Data[offset] % 3 == 0)) {
            try {
                auto empty_tensor = torch::empty({0}, input.options());
                auto result5 = torch::aminmax(empty_tensor);
            } catch (const std::exception&) {
                // Ignore exceptions from empty tensor
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
