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
        
        // Create input tensors A and B for lstsq
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data left for the second tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor B = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a driver value for rcond parameter
        double rcond = -1.0;
        if (offset + sizeof(float) <= Size) {
            float rcond_val;
            std::memcpy(&rcond_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Use the value from the input data, but ensure it's within a reasonable range
            if (std::isfinite(rcond_val)) {
                rcond = rcond_val;
            }
        }
        
        // Get a driver value for driver parameter
        std::string driver = "gels";
        if (offset < Size) {
            uint8_t driver_selector = Data[offset++];
            if (driver_selector % 2 == 0) {
                driver = "gels";
            } else {
                driver = "gelsy";
            }
        }
        
        // Call torch::lstsq with different combinations of parameters
        auto result = torch::lstsq(A, B, rcond);
        
        // Extract and use the results to ensure they're not optimized away
        auto solution = std::get<0>(result);
        auto qr = std::get<1>(result);
        
        // Perform some operations with the results to ensure they're not optimized away
        if (solution.defined()) {
            auto sum = solution.sum();
        }
        
        if (qr.defined()) {
            auto sum = qr.sum();
        }
        
        // Try another variant with default parameters
        auto default_result = torch::lstsq(A, B);
        
        // Try with different rcond values
        if (offset < Size) {
            uint8_t rcond_selector = Data[offset++];
            double new_rcond = (rcond_selector % 100) / 100.0;
            auto rcond_result = torch::lstsq(A, B, new_rcond);
        }
        
        // Try with explicit None for rcond
        auto none_rcond_result = torch::lstsq(A, B, c10::nullopt);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
