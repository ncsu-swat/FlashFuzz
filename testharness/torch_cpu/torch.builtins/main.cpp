#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a meaningful test
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test various tensor operations that are available in PyTorch C++
        
        // Test tensor type checking
        bool is_tensor = input.defined();
        
        // Test tensor properties
        auto dtype = input.dtype();
        auto device = input.device();
        
        // Test scalar conversion for scalar tensors
        if (input.numel() > 0 && input.dim() == 0) {
            try {
                if (input.dtype() == torch::kInt || input.dtype() == torch::kLong) {
                    auto int_val = input.item<int64_t>();
                }
                if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                    auto float_val = input.item<double>();
                }
                if (input.dtype() == torch::kBool) {
                    auto bool_val = input.item<bool>();
                }
            } catch (...) {
                // Conversion might fail for some tensor types
            }
        }
        
        // Test tensor size operations
        if (input.dim() > 0) {
            auto size_0 = input.size(0);
            auto sizes = input.sizes();
        }
        
        // Test tensor attributes
        auto shape = input.sizes();
        auto ndim = input.dim();
        auto numel = input.numel();
        
        // Test tensor range creation
        if (offset + 3 <= Size) {
            int64_t start = static_cast<int64_t>(Data[offset++]) % 10;
            int64_t end = start + (static_cast<int64_t>(Data[offset++]) % 10) + 1;
            int64_t step = (static_cast<int64_t>(Data[offset++]) % 5) + 1;
            
            auto range_result = torch::arange(start, end, step);
        }
        
        // Test tensor max and min operations
        if (input.numel() > 0) {
            try {
                auto max_val = torch::max(input);
                auto min_val = torch::min(input);
            } catch (...) {
                // May fail for complex types
            }
        }
        
        // Test tensor abs operation
        if (input.numel() > 0) {
            try {
                auto abs_val = torch::abs(input);
            } catch (...) {
                // May fail for some types
            }
        }
        
        // Test tensor divmod operation
        if (input.numel() > 0 && offset + 1 <= Size) {
            try {
                auto divisor = static_cast<int64_t>(Data[offset++]) + 1; // Avoid division by zero
                auto div_result = torch::div(input, torch::tensor(divisor));
                auto mod_result = torch::remainder(input, torch::tensor(divisor));
            } catch (...) {
                // May fail for some types
            }
        }
        
        // Test tensor round operation
        if (input.numel() > 0 && input.is_floating_point()) {
            try {
                auto round_result = torch::round(input);
            } catch (...) {
                // May fail for some types
            }
        }
        
        // Test tensor sum operation
        if (input.numel() > 0) {
            try {
                auto sum_result = torch::sum(input);
            } catch (...) {
                // May fail for some types
            }
        }
        
        // Test tensor cloning and copying
        auto cloned = input.clone();
        auto copied = input.detach();
        
        // Test tensor reshaping if possible
        if (input.numel() > 0) {
            try {
                auto flattened = input.flatten();
            } catch (...) {
                // May fail for some configurations
            }
        }
        
        // Test tensor comparison operations
        if (input.numel() > 0) {
            try {
                auto eq_result = torch::eq(input, input);
                auto ne_result = torch::ne(input, input);
            } catch (...) {
                // May fail for some types
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