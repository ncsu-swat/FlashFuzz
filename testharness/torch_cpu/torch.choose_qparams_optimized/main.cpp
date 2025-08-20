#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for choose_qparams_optimized
        uint8_t numel_byte = 0;
        if (offset < Size) {
            numel_byte = Data[offset++];
        }
        int64_t numel = (numel_byte % 100) + 1; // Range: 1-100
        
        uint8_t n_bins_byte = 0;
        if (offset < Size) {
            n_bins_byte = Data[offset++];
        }
        int64_t n_bins = (n_bins_byte % 256) + 1; // Range: 1-256
        
        uint8_t ratio_byte = 0;
        if (offset < Size) {
            ratio_byte = Data[offset++];
        }
        double ratio = (ratio_byte / 255.0) * 2.0; // Range: 0.0-2.0
        
        uint8_t bit_width_byte = 0;
        if (offset < Size) {
            bit_width_byte = Data[offset++];
        }
        int64_t bit_width = (bit_width_byte % 8) + 1; // Range: 1-8 bits
        
        // Apply the operation
        auto result = torch::choose_qparams_optimized(
            input_tensor,
            numel,
            n_bins,
            ratio,
            bit_width
        );
        
        // Access the result to ensure it's computed
        auto scale = std::get<0>(result);
        auto zero_point = std::get<1>(result);
        
        // Optionally, perform some operation with the results to ensure they're valid
        if (scale.defined() && zero_point.defined()) {
            auto dummy = scale + 1.0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}