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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.trunc operation
        torch::Tensor result = torch::trunc(input_tensor);
        
        // Try in-place version if there's enough data to decide
        if (offset < Size) {
            bool use_inplace = Data[offset++] % 2 == 0;
            if (use_inplace) {
                torch::Tensor input_copy = input_tensor.clone();
                input_copy.trunc_();
            }
        }
        
        // Try with different output tensor if there's enough data
        if (offset < Size) {
            bool use_out = Data[offset++] % 2 == 0;
            if (use_out) {
                torch::Tensor output = torch::empty_like(input_tensor);
                torch::trunc_out(output, input_tensor);
            }
        }
        
        // Try with named tensor if there's enough data
        if (offset < Size && input_tensor.dim() > 0) {
            bool use_named = Data[offset++] % 2 == 0;
            if (use_named) {
                std::vector<torch::Dimname> names;
                for (int64_t i = 0; i < input_tensor.dim(); i++) {
                    names.push_back(torch::Dimname::fromSymbol(c10::Symbol::dimname("dim" + std::to_string(i))));
                }
                torch::Tensor named_input = input_tensor.refine_names(names);
                torch::Tensor named_result = torch::trunc(named_input);
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
