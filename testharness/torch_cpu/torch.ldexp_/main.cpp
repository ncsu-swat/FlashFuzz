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
        
        // Create the first tensor (x)
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the second tensor (exponent)
        // If we have enough data left, create another tensor for exponent
        torch::Tensor exponent;
        if (offset < Size) {
            exponent = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure exponent has the same shape as x or can be broadcast
            if (exponent.dim() > x.dim()) {
                // Reshape exponent to match x's shape if possible
                std::vector<int64_t> new_shape;
                for (int i = 0; i < x.dim(); i++) {
                    new_shape.push_back(x.size(i));
                }
                
                try {
                    exponent = exponent.reshape(new_shape);
                } catch (...) {
                    // If reshape fails, create a new tensor with the same shape as x
                    exponent = torch::ones_like(x).to(torch::kInt32);
                }
            }
        } else {
            // If we don't have enough data, create a simple exponent tensor
            exponent = torch::ones_like(x).to(torch::kInt32);
        }
        
        // Convert tensors to supported types for ldexp if needed
        if (!x.is_floating_point() && !x.is_complex()) {
            x = x.to(torch::kFloat32);
        }
        
        if (exponent.dtype() != torch::kInt32 && exponent.dtype() != torch::kInt64) {
            exponent = exponent.to(torch::kInt32);
        }
        
        // Make a copy of x to test the in-place operation
        torch::Tensor x_copy = x.clone();
        
        // Apply ldexp_ (in-place operation)
        try {
            x_copy.ldexp_(exponent);
        } catch (...) {
            // If in-place fails, try the out-of-place version
            try {
                torch::Tensor result = torch::ldexp(x, exponent);
            } catch (...) {
                // Both versions failed, but that's okay for fuzzing
            }
        }
        
        // Try with scalar exponent as tensor
        try {
            int scalar_exp = 0;
            if (offset + sizeof(int) <= Size) {
                std::memcpy(&scalar_exp, Data + offset, sizeof(int));
                offset += sizeof(int);
            }
            
            torch::Tensor x_scalar_copy = x.clone();
            torch::Tensor scalar_tensor = torch::tensor(scalar_exp);
            x_scalar_copy.ldexp_(scalar_tensor);
        } catch (...) {
            // Scalar version failed, but that's okay for fuzzing
        }
        
        // Try with different dtypes
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                torch::Tensor x_dtype = x.to(dtype);
                if (x_dtype.is_floating_point() || x_dtype.is_complex()) {
                    torch::Tensor x_dtype_copy = x_dtype.clone();
                    x_dtype_copy.ldexp_(exponent);
                }
            } catch (...) {
                // Type conversion or operation failed, but that's okay for fuzzing
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
