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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create hidden state tensor
        torch::Tensor hx = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensors
        torch::Tensor w_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor w_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create bias tensors
        torch::Tensor b_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor b_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create packed weight tensors
        torch::Tensor packed_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor packed_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create column offset tensors
        torch::Tensor col_offsets_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor col_offsets_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse scale parameters
        double scale_ih = 1.0;
        double scale_hh = 1.0;
        int64_t zero_point_ih = 0;
        int64_t zero_point_hh = 0;
        
        if (offset + 2 * sizeof(double) + 2 * sizeof(int64_t) <= Size) {
            std::memcpy(&scale_ih, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            std::memcpy(&scale_hh, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            std::memcpy(&zero_point_ih, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&zero_point_hh, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure scales are positive and not too extreme
        scale_ih = std::abs(scale_ih);
        if (scale_ih < 1e-6) scale_ih = 1e-6;
        if (scale_ih > 1e6) scale_ih = 1e6;
        
        scale_hh = std::abs(scale_hh);
        if (scale_hh < 1e-6) scale_hh = 1e-6;
        if (scale_hh > 1e6) scale_hh = 1e6;
        
        // Limit zero_points to reasonable range
        zero_point_ih = zero_point_ih % 256;
        zero_point_hh = zero_point_hh % 256;
        
        // Call quantized_rnn_relu_cell with all required parameters
        auto result = torch::quantized_rnn_relu_cell(
            input, hx, w_ih, w_hh, b_ih, b_hh, 
            packed_ih, packed_hh, col_offsets_ih, col_offsets_hh,
            scale_ih, scale_hh, zero_point_ih, zero_point_hh);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}