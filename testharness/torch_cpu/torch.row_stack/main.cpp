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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Determine number of tensors to create (1-4)
        uint8_t num_tensors = (Size > 0) ? (Data[0] % 4) + 1 : 1;
        offset++;
        
        // Create a vector to hold our tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors
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
        
        // Apply row_stack operation (equivalent to vstack/vertical stack)
        try {
            torch::Tensor result = torch::row_stack(tensors);
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and part of testing
            return 0;
        }
        
        // Try alternative syntax with TensorList
        try {
            torch::Tensor result = torch::vstack(tensors);
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and part of testing
            return 0;
        }
        
        // Try with empty tensor list if we have enough data
        if (offset < Size) {
            try {
                std::vector<torch::Tensor> empty_tensors;
                torch::Tensor result = torch::row_stack(empty_tensors);
            } catch (const c10::Error& e) {
                // Expected exception
            }
        }
        
        // Try with tensors of different dtypes if we have multiple tensors
        if (tensors.size() > 1 && offset < Size) {
            try {
                // Create a tensor with a different dtype
                uint8_t dtype_selector = Data[offset++];
                torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
                
                // Convert one tensor to the new dtype
                tensors[0] = tensors[0].to(dtype);
                
                // Try row_stack with mixed dtypes
                torch::Tensor result = torch::row_stack(tensors);
            } catch (const c10::Error& e) {
                // Expected exception
            }
        }
        
        // Try with tensors that have incompatible shapes
        if (offset + 4 < Size) {
            try {
                // Create tensors with potentially incompatible shapes
                torch::Tensor t1 = torch::ones({2, 3});
                torch::Tensor t2 = torch::ones({3, 4});
                
                std::vector<torch::Tensor> incompatible_tensors = {t1, t2};
                torch::Tensor result = torch::row_stack(incompatible_tensors);
            } catch (const c10::Error& e) {
                // Expected exception
            }
        }
        
        // Try with scalar tensors
        if (offset < Size) {
            try {
                std::vector<torch::Tensor> scalar_tensors;
                scalar_tensors.push_back(torch::tensor(1.0));
                scalar_tensors.push_back(torch::tensor(2.0));
                
                torch::Tensor result = torch::row_stack(scalar_tensors);
            } catch (const c10::Error& e) {
                // Expected exception
            }
        }
        
        // Try with 1D tensors
        if (offset < Size) {
            try {
                std::vector<torch::Tensor> vector_tensors;
                vector_tensors.push_back(torch::ones({3}));
                vector_tensors.push_back(torch::ones({3}));
                
                torch::Tensor result = torch::row_stack(vector_tensors);
            } catch (const c10::Error& e) {
                // Expected exception
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
