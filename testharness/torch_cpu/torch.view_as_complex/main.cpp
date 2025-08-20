#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // view_as_complex requires a tensor with a last dimension of size 2
        // and a floating-point or complex dtype
        if (input_tensor.dim() > 0 && 
            input_tensor.size(-1) == 2 && 
            (input_tensor.dtype() == torch::kFloat || 
             input_tensor.dtype() == torch::kDouble || 
             input_tensor.dtype() == torch::kHalf || 
             input_tensor.dtype() == torch::kBFloat16)) {
            
            // Apply view_as_complex operation
            torch::Tensor result = torch::view_as_complex(input_tensor);
            
            // Perform some operations on the result to ensure it's used
            if (result.numel() > 0) {
                auto sum = result.sum();
                auto mean = result.mean();
            }
        }
        
        // Try with different tensor configurations if there's more data
        if (offset + 2 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to reshape the tensor to have last dimension of size 2 if possible
            if (another_tensor.numel() >= 2 && another_tensor.numel() % 2 == 0) {
                std::vector<int64_t> new_shape;
                
                if (another_tensor.dim() > 1) {
                    // Keep all dimensions except the last one
                    for (int i = 0; i < another_tensor.dim() - 1; i++) {
                        new_shape.push_back(another_tensor.size(i));
                    }
                    
                    // Adjust the last dimension
                    int64_t last_dim_size = another_tensor.size(-1);
                    int64_t second_last_dim = another_tensor.size(-2);
                    
                    new_shape.push_back(second_last_dim * last_dim_size / 2);
                    new_shape.push_back(2);
                } else {
                    // For 1D tensor, reshape to [n/2, 2]
                    new_shape.push_back(another_tensor.numel() / 2);
                    new_shape.push_back(2);
                }
                
                // Try to reshape and apply view_as_complex
                try {
                    auto reshaped = another_tensor.reshape(new_shape);
                    
                    // Convert to a compatible dtype if needed
                    torch::Tensor compatible_tensor;
                    if (reshaped.dtype() == torch::kFloat || 
                        reshaped.dtype() == torch::kDouble || 
                        reshaped.dtype() == torch::kHalf || 
                        reshaped.dtype() == torch::kBFloat16) {
                        compatible_tensor = reshaped;
                    } else {
                        compatible_tensor = reshaped.to(torch::kFloat);
                    }
                    
                    torch::Tensor complex_result = torch::view_as_complex(compatible_tensor);
                    
                    // Use the result
                    if (complex_result.numel() > 0) {
                        auto abs_val = torch::abs(complex_result);
                    }
                } catch (const std::exception&) {
                    // Reshape or view_as_complex might fail, which is fine for fuzzing
                }
            }
        }
        
        // Try with a tensor that has exactly the right shape for view_as_complex
        if (offset + 2 < Size) {
            uint8_t dtype_byte = Data[offset++];
            torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_byte);
            
            // Make sure we have a floating point type
            if (dtype != torch::kFloat && dtype != torch::kDouble && 
                dtype != torch::kHalf && dtype != torch::kBFloat16) {
                dtype = torch::kFloat;
            }
            
            // Create a shape with last dimension 2
            uint8_t rank_byte = Data[offset++];
            uint8_t rank = fuzzer_utils::parseRank(rank_byte);
            
            // Ensure rank is at least 1
            if (rank == 0) rank = 1;
            
            std::vector<int64_t> shape;
            for (int i = 0; i < rank - 1; i++) {
                if (offset < Size) {
                    shape.push_back(1 + (Data[offset++] % 5)); // Small dimensions 1-5
                } else {
                    shape.push_back(1);
                }
            }
            shape.push_back(2); // Last dimension must be 2
            
            // Create tensor with the right shape and dtype
            torch::Tensor special_tensor = torch::rand(shape, torch::TensorOptions().dtype(dtype));
            
            // Apply view_as_complex
            torch::Tensor complex_view = torch::view_as_complex(special_tensor);
            
            // Use the result
            auto real_part = torch::real(complex_view);
            auto imag_part = torch::imag(complex_view);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}