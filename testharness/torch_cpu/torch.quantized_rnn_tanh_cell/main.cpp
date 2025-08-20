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
        
        // Create hidden state tensor
        torch::Tensor hx = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensor
        torch::Tensor w_ih = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create recurrent weight tensor
        torch::Tensor w_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create bias tensor (optional)
        torch::Tensor b_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor b_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create packed tensors
        torch::Tensor packed_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor packed_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create column offset tensors
        torch::Tensor col_offsets_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor col_offsets_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse scale parameters for quantization
        double scale_ih = 1.0;
        double scale_hh = 1.0;
        int64_t zero_point_ih = 0;
        int64_t zero_point_hh = 0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale_ih, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale_ih = std::abs(scale_ih) + 1e-6; // Ensure positive scale
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scale_hh, Data + offset, sizeof(double));
            offset += sizeof(double);
            scale_hh = std::abs(scale_hh) + 1e-6; // Ensure positive scale
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point_ih, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&zero_point_hh, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Try to call the quantized_rnn_tanh_cell function
        torch::Tensor result = torch::quantized_rnn_tanh_cell(
            input, hx, w_ih, w_hh, b_ih, b_hh,
            packed_ih, packed_hh, col_offsets_ih, col_offsets_hh,
            scale_ih, scale_hh, zero_point_ih, zero_point_hh
        );
        
        // Perform some operations on the result to ensure it's used
        if (result.defined()) {
            auto sum = result.sum();
            if (sum.numel() > 0) {
                volatile double val = sum.item<double>();
                (void)val;
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