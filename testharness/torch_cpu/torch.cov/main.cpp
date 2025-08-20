#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for cov function
        int64_t correction = 1;
        torch::Tensor fweights;
        torch::Tensor aweights;
        
        // Use remaining bytes to determine parameters if available
        if (offset + 1 < Size) {
            correction = Data[offset++] & 0x1;
        }
        
        // Optionally create fweights tensor
        bool use_fweights = false;
        if (offset + 1 < Size) {
            use_fweights = Data[offset++] & 0x1;
            
            if (use_fweights && offset < Size) {
                try {
                    fweights = fuzzer_utils::createTensor(Data, Size, offset);
                } catch (const std::exception&) {
                    // If fweights tensor creation fails, proceed without it
                    use_fweights = false;
                }
            }
        }
        
        // Optionally create aweights tensor
        bool use_aweights = false;
        if (offset + 1 < Size) {
            use_aweights = Data[offset++] & 0x1;
            
            if (use_aweights && offset < Size) {
                try {
                    aweights = fuzzer_utils::createTensor(Data, Size, offset);
                } catch (const std::exception&) {
                    // If aweights tensor creation fails, proceed without it
                    use_aweights = false;
                }
            }
        }
        
        // Call torch::cov with different parameter combinations
        torch::Tensor result;
        
        if (use_fweights && use_aweights) {
            result = torch::cov(input, correction, fweights, aweights);
        } else if (use_fweights) {
            result = torch::cov(input, correction, fweights);
        } else {
            result = torch::cov(input, correction);
        }
        
        // Try to access result to ensure computation is performed
        if (result.defined()) {
            auto sizes = result.sizes();
            auto numel = result.numel();
            
            // Force evaluation of the tensor
            if (numel > 0) {
                auto item = result.item();
            }
        }
        
        // Try alternative parameter combinations if there's enough data
        if (offset + 1 < Size) {
            int64_t alt_correction = Data[offset++] & 0x1;
            
            // Call with alternative parameters
            torch::Tensor alt_result = torch::cov(input, alt_correction);
            
            // Force evaluation
            if (alt_result.defined() && alt_result.numel() > 0) {
                auto item = alt_result.item();
            }
        }
        
        // Try with freeform values
        if (offset + 1 < Size) {
            int64_t freeform_correction = Data[offset++] & 0x1;
            
            torch::Tensor freeform_result = torch::cov(input, freeform_correction);
            
            if (freeform_result.defined() && freeform_result.numel() > 0) {
                auto item = freeform_result.item();
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