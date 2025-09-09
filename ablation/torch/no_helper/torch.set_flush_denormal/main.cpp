#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>  // PyTorch C++ frontend

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the boolean mode
        if (Size < 1) {
            return 0;
        }
        
        // Extract boolean mode from fuzzer input
        bool mode = (Data[offset] % 2) == 1;
        offset++;
        
        // Test basic functionality
        bool result = torch::set_flush_denormal(mode);
        
        // Test multiple calls with same mode
        bool result2 = torch::set_flush_denormal(mode);
        
        // Test toggling mode if we have more data
        if (offset < Size) {
            bool toggle_mode = (Data[offset] % 2) == 1;
            torch::set_flush_denormal(toggle_mode);
            
            // Toggle back
            torch::set_flush_denormal(mode);
        }
        
        // Test with extreme values and edge cases
        torch::set_flush_denormal(true);
        torch::set_flush_denormal(false);
        
        // Test rapid toggling if we have enough data
        if (Size >= 10) {
            for (size_t i = 1; i < std::min(Size, size_t(10)); ++i) {
                bool rapid_mode = (Data[i] % 2) == 1;
                torch::set_flush_denormal(rapid_mode);
            }
        }
        
        // Test creating tensors with denormal values to verify behavior
        try {
            // Create a very small float64 tensor that might be denormal
            auto tensor_f64 = torch::tensor({1e-323}, torch::dtype(torch::kFloat64));
            
            // Create a very small float32 tensor that might be denormal  
            auto tensor_f32 = torch::tensor({1e-40f}, torch::dtype(torch::kFloat32));
            
            // Test with different modes
            torch::set_flush_denormal(true);
            auto tensor_flush_true = torch::tensor({1e-323}, torch::dtype(torch::kFloat64));
            
            torch::set_flush_denormal(false);
            auto tensor_flush_false = torch::tensor({1e-323}, torch::dtype(torch::kFloat64));
            
        } catch (const std::exception& tensor_e) {
            // Tensor creation might fail, but that's okay for fuzzing
        }
        
        // Test state consistency - set a known state at the end
        torch::set_flush_denormal(false);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}