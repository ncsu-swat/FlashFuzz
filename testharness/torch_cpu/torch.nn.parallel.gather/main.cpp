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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse parameters for gather operation
        uint8_t dim_byte = 0;
        if (offset < Size) {
            dim_byte = Data[offset++];
        }
        
        // Determine dimension for gather operation
        int64_t dim = dim_byte % (std::max(1, static_cast<int>(input_tensor.dim())));
        
        // Create a list of tensors to gather
        std::vector<torch::Tensor> tensor_list;
        
        // Determine number of tensors to create (1-4)
        uint8_t num_tensors = 1;
        if (offset < Size) {
            num_tensors = (Data[offset++] % 4) + 1;
        }
        
        // Create tensors for the list
        for (uint8_t i = 0; i < num_tensors; i++) {
            if (offset < Size) {
                torch::Tensor t = fuzzer_utils::createTensor(Data, Size, offset);
                tensor_list.push_back(t);
            } else {
                // If we run out of data, just duplicate the input tensor
                tensor_list.push_back(input_tensor);
            }
        }
        
        // Parse output_size parameter
        int64_t output_size = 0;
        if (offset < Size) {
            uint8_t output_size_byte = Data[offset++];
            output_size = output_size_byte % 100; // Limit to reasonable size
        }
        
        // Try different variants of gather
        
        // Variant 1: Basic gather using torch::cat
        try {
            auto result1 = torch::cat(tensor_list, dim);
        } catch (const c10::Error &e) {
            // PyTorch specific exceptions are expected and handled
        }
        
        // Variant 2: Gather with index tensor
        try {
            if (!tensor_list.empty()) {
                auto index_tensor = torch::randint(0, tensor_list[0].size(dim), {output_size});
                auto result2 = torch::gather(tensor_list[0], dim, index_tensor);
            }
        } catch (const c10::Error &e) {
            // PyTorch specific exceptions are expected and handled
        }
        
        // Variant 3: Stack tensors
        try {
            auto result3 = torch::stack(tensor_list, dim);
        } catch (const c10::Error &e) {
            // PyTorch specific exceptions are expected and handled
        }
        
        // Variant 4: Edge case - empty tensor list
        try {
            std::vector<torch::Tensor> empty_list;
            if (!empty_list.empty()) {
                auto result4 = torch::cat(empty_list, dim);
            }
        } catch (const c10::Error &e) {
            // PyTorch specific exceptions are expected and handled
        }
        
        // Variant 5: Edge case - negative dimension
        try {
            if (!tensor_list.empty()) {
                auto result5 = torch::cat(tensor_list, -dim - 1);
            }
        } catch (const c10::Error &e) {
            // PyTorch specific exceptions are expected and handled
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
