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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor y
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create optional x tensor if we have enough data left
        torch::Tensor x;
        bool has_x = false;
        if (offset + 4 < Size) {
            x = fuzzer_utils::createTensor(Data, Size, offset);
            has_x = true;
        }
        
        // Get dim parameter if we have data left
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get dx parameter if we have data left
        double dx = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&dx, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Try different variants of trapezoid
        if (has_x) {
            // Variant 1: trapezoid with x and y tensors
            torch::Tensor result1 = torch::trapezoid(y, x, dim);
            
            // Variant 2: trapezoid with x, y, and dim
            torch::Tensor result2 = torch::trapezoid(y, x);
        } else {
            // Variant 3: trapezoid with y and dx
            torch::Tensor result3 = torch::trapezoid(y, dx, dim);
            
            // Variant 4: trapezoid with y and dx (no dim)
            torch::Tensor result4 = torch::trapezoid(y, dx);
            
            // Variant 5: trapezoid with just y
            torch::Tensor result5 = torch::trapezoid(y);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
