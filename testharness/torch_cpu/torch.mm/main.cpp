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
        
        // Create first input tensor for matrix multiplication
        torch::Tensor mat1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor for matrix multiplication
        if (offset < Size) {
            torch::Tensor mat2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Attempt to perform matrix multiplication
            // torch.mm performs matrix multiplication between two 2D tensors
            // If tensors are not 2D, we'll let PyTorch handle the error
            torch::Tensor result = torch::mm(mat1, mat2);
            
            // Optional: Try to access result to ensure computation is performed
            if (result.defined()) {
                auto sizes = result.sizes();
                auto numel = result.numel();
                
                // Force evaluation of the tensor
                if (numel > 0) {
                    auto item = result.item();
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
