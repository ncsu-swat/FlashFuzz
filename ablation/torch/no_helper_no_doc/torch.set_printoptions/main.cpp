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
        
        // Need at least some bytes to work with
        if (Size < 16) {
            return 0;
        }

        // Extract fuzzing parameters
        int precision = extract_int(Data, Size, offset) % 20; // 0-19
        int threshold = extract_int(Data, Size, offset) % 10000; // 0-9999
        int edgeitems = extract_int(Data, Size, offset) % 10; // 0-9
        int linewidth = extract_int(Data, Size, offset) % 1000 + 1; // 1-1000
        bool profile = extract_bool(Data, Size, offset);
        bool sci_mode = extract_bool(Data, Size, offset);
        
        // Test basic set_printoptions with precision
        torch::set_printoptions(precision);
        
        // Test with threshold
        torch::set_printoptions(/*precision=*/precision, /*threshold=*/threshold);
        
        // Test with edgeitems
        torch::set_printoptions(/*precision=*/precision, /*threshold=*/threshold, /*edgeitems=*/edgeitems);
        
        // Test with linewidth
        torch::set_printoptions(/*precision=*/precision, /*threshold=*/threshold, /*edgeitems=*/edgeitems, /*linewidth=*/linewidth);
        
        // Test with profile
        torch::set_printoptions(/*precision=*/precision, /*threshold=*/threshold, /*edgeitems=*/edgeitems, /*linewidth=*/linewidth, /*profile=*/profile);
        
        // Test with sci_mode
        torch::set_printoptions(/*precision=*/precision, /*threshold=*/threshold, /*edgeitems=*/edgeitems, /*linewidth=*/linewidth, /*profile=*/profile, /*sci_mode=*/sci_mode);
        
        // Test edge cases
        // Negative precision (should be handled gracefully)
        torch::set_printoptions(-1);
        
        // Zero values
        torch::set_printoptions(0, 0, 0, 1);
        
        // Very large values
        torch::set_printoptions(100, 100000, 50, 10000);
        
        // Create a tensor to verify the print options actually work
        auto tensor = torch::randn({3, 3});
        std::ostringstream oss;
        oss << tensor;
        
        // Test with different tensor types and sizes to ensure print options work
        auto int_tensor = torch::randint(0, 100, {2, 2}, torch::kInt);
        oss << int_tensor;
        
        auto large_tensor = torch::randn({10, 10});
        oss << large_tensor;
        
        // Test boolean tensors
        auto bool_tensor = torch::randint(0, 2, {3, 3}, torch::kBool);
        oss << bool_tensor;
        
        // Reset to default values periodically
        if (extract_bool(Data, Size, offset)) {
            torch::set_printoptions(4, 1000, 3, 80, false, false);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}