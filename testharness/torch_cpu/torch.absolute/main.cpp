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
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.absolute operation
        torch::Tensor result = torch::abs(input_tensor);
        
        // Try alternative syntax
        torch::Tensor result2 = input_tensor.abs();
        
        // Try in-place version if possible
        if (input_tensor.is_floating_point() || input_tensor.is_complex()) {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.abs_();
        }
        
        // Try with named tensor if we have enough data
        if (offset + 1 < Size && input_tensor.dim() > 0) {
            // Get a name for the first dimension
            std::vector<torch::Dimname> names;
            for (int64_t i = 0; i < input_tensor.dim(); i++) {
                std::string dim_name = "dim" + std::to_string(i);
                names.push_back(torch::Dimname::fromSymbol(torch::Symbol::dimname(dim_name)));
            }
            
            auto named_tensor = input_tensor.refine_names(names);
            auto named_result = torch::abs(named_tensor);
        }
        
        // Try with out parameter if we have enough data
        if (offset + 1 < Size) {
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::abs_out(out_tensor, input_tensor);
        }
        
        // Try with different scalar types if input is integer
        if (input_tensor.scalar_type() == torch::kInt || input_tensor.scalar_type() == torch::kLong) {
            torch::Tensor result_converted = torch::abs(input_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
