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
        
        // Need at least 1 byte to determine number of tensors
        if (Size < 1) {
            return 0;
        }
        
        // Determine number of tensors to concatenate (1-8)
        uint8_t num_tensors = (Data[offset++] % 8) + 1;
        
        // Create a vector to hold our tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors with various properties
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception& e) {
                // If we can't create a tensor, just continue with what we have
                break;
            }
        }
        
        // Need at least one tensor to proceed
        if (tensors.empty()) {
            return 0;
        }
        
        // Determine dimension to concatenate along
        int64_t dim = 0;
        if (offset < Size) {
            // Get a dimension value from the input data
            // Allow negative dimensions to test edge cases
            int8_t dim_value;
            std::memcpy(&dim_value, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            
            // If the tensor has dimensions, use the input to select one
            if (!tensors[0].sizes().empty()) {
                // Allow negative indexing (e.g., -1 for last dimension)
                dim = dim_value;
            }
        }
        
        // Apply torch.cat operation
        try {
            torch::Tensor result = torch::cat(tensors, dim);
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected for invalid inputs
            return 0;
        }
        
        // Try another variant with named arguments
        try {
            torch::Tensor result = torch::cat({tensors}, dim);
        } catch (const c10::Error& e) {
            return 0;
        }
        
        // Try with empty tensor list if we have enough data
        if (offset < Size) {
            try {
                std::vector<torch::Tensor> empty_tensors;
                torch::Tensor result = torch::cat(empty_tensors, 0);
            } catch (const c10::Error& e) {
                // Expected to fail
            }
        }
        
        // Try with tensors of different dtypes if we have multiple tensors
        if (tensors.size() > 1 && offset < Size) {
            try {
                // Convert the second tensor to a different dtype
                uint8_t dtype_selector = Data[offset++];
                auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                tensors[1] = tensors[1].to(dtype);
                
                torch::Tensor result = torch::cat(tensors, dim);
            } catch (const c10::Error& e) {
                // Expected to fail in some cases
            }
        }
        
        // Try with out parameter if we have enough data
        if (offset < Size && !tensors.empty()) {
            try {
                // Create an output tensor with the same dtype as the first tensor
                auto options = torch::TensorOptions().dtype(tensors[0].dtype());
                torch::Tensor out = torch::empty({1}, options);
                
                // Use the out variant of cat
                torch::cat_out(out, tensors, dim);
            } catch (const c10::Error& e) {
                // Expected to fail in some cases
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
