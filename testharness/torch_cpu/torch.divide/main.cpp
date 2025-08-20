#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset < Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply torch.divide operation
            torch::Tensor result = torch::divide(input1, input2);
            
            // Try different variants of divide
            if (offset + 1 < Size) {
                uint8_t variant = Data[offset++];
                
                // Test out_variant
                torch::Tensor out = torch::empty_like(input1);
                torch::divide_out(out, input1, input2);
                
                // Test scalar variants if we have more data
                if (offset < Size) {
                    double scalar_value = static_cast<double>(Data[offset++]);
                    
                    // Test tensor / scalar
                    torch::Tensor result_scalar = torch::divide(input1, scalar_value);
                    
                    // Test scalar / tensor - create scalar tensor first
                    torch::Tensor scalar_tensor = torch::scalar_tensor(scalar_value, input1.options());
                    torch::Tensor result_scalar_first = torch::divide(scalar_tensor, input1);
                }
            }
        } else {
            // If we don't have enough data for a second tensor, try scalar division
            if (offset < Size) {
                double scalar_value = static_cast<double>(Data[offset++]);
                torch::Tensor result = torch::divide(input1, scalar_value);
            }
        }
        
        // Try inplace division if we have more data
        if (offset < Size) {
            uint8_t inplace_flag = Data[offset++];
            if (inplace_flag % 2 == 0) {
                // Create a copy to avoid modifying the original tensor
                torch::Tensor input_copy = input1.clone();
                
                // Check if we have enough data for a second tensor
                if (offset < Size) {
                    torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
                    input_copy.divide_(input2);
                } else if (offset < Size) {
                    // Try scalar inplace division
                    double scalar_value = static_cast<double>(Data[offset++]);
                    input_copy.divide_(scalar_value);
                }
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