#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Use first byte to select an integral dtype (bitwise_not requires integral types)
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType dtype;
        switch (dtype_selector % 5) {
            case 0: dtype = torch::kBool; break;
            case 1: dtype = torch::kInt8; break;
            case 2: dtype = torch::kInt16; break;
            case 3: dtype = torch::kInt32; break;
            case 4: dtype = torch::kInt64; break;
            default: dtype = torch::kInt32; break;
        }
        
        // Create input tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to integral type for bitwise operations
        input_tensor = input_tensor.to(dtype);
        
        // Apply bitwise_not operation
        torch::Tensor result = torch::bitwise_not(input_tensor);
        
        // Try inplace version
        if (offset < Size) {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.bitwise_not_();
        }
        
        // Try with out parameter
        if (offset < Size) {
            torch::Tensor out_tensor = torch::empty_like(input_tensor);
            torch::bitwise_not_out(out_tensor, input_tensor);
        }
        
        // Try with different integral dtypes
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Select another integral dtype
            torch::ScalarType new_dtype;
            switch (option_byte % 5) {
                case 0: new_dtype = torch::kBool; break;
                case 1: new_dtype = torch::kInt8; break;
                case 2: new_dtype = torch::kInt16; break;
                case 3: new_dtype = torch::kInt32; break;
                case 4: new_dtype = torch::kInt64; break;
                default: new_dtype = torch::kInt32; break;
            }
            
            try {
                torch::Tensor converted = input_tensor.to(new_dtype);
                torch::Tensor result2 = torch::bitwise_not(converted);
            } catch (const std::exception&) {
                // Some conversions may have issues
            }
        }
        
        // Test with different tensor shapes
        if (offset + 4 < Size) {
            try {
                // Create a multi-dimensional tensor
                int64_t dim1 = (Data[offset++] % 8) + 1;
                int64_t dim2 = (Data[offset++] % 8) + 1;
                
                torch::Tensor multi_dim = torch::randint(0, 256, {dim1, dim2}, torch::dtype(torch::kInt32));
                torch::Tensor multi_result = torch::bitwise_not(multi_dim);
                
                // Test with contiguous and non-contiguous tensors
                torch::Tensor transposed = multi_dim.t();
                torch::Tensor trans_result = torch::bitwise_not(transposed);
            } catch (const std::exception&) {
                // Shape operations may fail
            }
        }
        
        // Test with scalar tensor
        if (offset < Size) {
            try {
                int64_t scalar_val = static_cast<int64_t>(Data[offset++]);
                torch::Tensor scalar_tensor = torch::tensor(scalar_val, torch::dtype(torch::kInt64));
                torch::Tensor scalar_result = torch::bitwise_not(scalar_tensor);
            } catch (const std::exception&) {
                // Scalar operations may have issues
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}