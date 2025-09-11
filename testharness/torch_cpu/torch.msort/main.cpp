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
        
        // Apply msort operation
        torch::Tensor result = torch::msort(input);
        
        // Try with different tensor types if we have more data
        if (offset + 1 < Size) {
            // Create a 2D tensor for more complex sorting
            std::vector<int64_t> shape = {3, 4};
            torch::Tensor tensor_2d = torch::randn(shape);
            torch::Tensor result_2d = torch::msort(tensor_2d);
        }
        
        // Try with named dimension tensor
        if (input.dim() > 0 && offset < Size) {
            // Create a named tensor by adding names to the input tensor
            std::vector<torch::Dimname> names;
            for (int64_t i = 0; i < input.dim(); i++) {
                names.push_back(torch::Dimname::fromSymbol(torch::Symbol::dimname("dim" + std::to_string(i))));
            }
            
            auto named_input = input.refine_names(names);
            
            // Sort the named tensor
            auto result_named = torch::msort(named_input);
        }
        
        // Try with empty tensor
        if (offset < Size) {
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_result = torch::msort(empty_tensor);
        }
        
        // Try with scalar tensor
        if (offset < Size) {
            torch::Tensor scalar_tensor = torch::tensor(static_cast<float>(Data[offset++]));
            torch::Tensor scalar_result = torch::msort(scalar_tensor);
        }
        
        // Try with boolean tensor
        if (offset < Size) {
            std::vector<int64_t> shape = {2, 3};
            auto options = torch::TensorOptions().dtype(torch::kBool);
            torch::Tensor bool_tensor = torch::empty(shape, options);
            torch::Tensor bool_result = torch::msort(bool_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
