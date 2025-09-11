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
        
        // Create input tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.conj operation
        torch::Tensor result = torch::conj(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto item = result.item();
        }
        
        // Try some additional operations with the result
        if (result.defined()) {
            // Test conjugate view
            torch::Tensor conj_view = input_tensor.conj();
            
            // Test conjugate physical copy
            torch::Tensor conj_copy = torch::conj_physical(input_tensor);
            
            // Test is_conj flag
            bool is_conj = result.is_conj();
            
            // Test resolve_conj
            torch::Tensor resolved = torch::resolve_conj(result);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
