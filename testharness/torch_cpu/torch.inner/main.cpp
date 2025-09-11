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
        
        // Create two input tensors for inner product
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            // Try the operation with just one tensor (against itself)
            torch::Tensor result = torch::inner(tensor1, tensor1);
            return 0;
        }
        
        torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.inner operation
        // inner(input, other, *, out=None) -> Tensor
        torch::Tensor result = torch::inner(tensor1, tensor2);
        
        // Try different variants if we have more data
        if (offset + 1 < Size) {
            uint8_t variant = Data[offset++];
            
            // Try with output tensor
            if (variant % 3 == 0) {
                // Create an output tensor with appropriate shape
                torch::Tensor output;
                try {
                    // Try to compute the expected output shape
                    std::vector<int64_t> output_shape;
                    if (tensor1.dim() > 1) {
                        for (int64_t i = 0; i < tensor1.dim() - 1; i++) {
                            output_shape.push_back(tensor1.size(i));
                        }
                    }
                    if (tensor2.dim() > 1) {
                        for (int64_t i = 0; i < tensor2.dim() - 1; i++) {
                            output_shape.push_back(tensor2.size(i));
                        }
                    }
                    
                    // Create output tensor with appropriate shape and dtype
                    output = torch::empty(output_shape, tensor1.options());
                    
                    // Call inner with output tensor
                    torch::inner_out(output, tensor1, tensor2);
                } catch (const std::exception&) {
                    // If shape calculation fails, just continue
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
