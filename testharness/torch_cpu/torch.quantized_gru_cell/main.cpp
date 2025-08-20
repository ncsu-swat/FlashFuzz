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
        
        // Create weight_ih tensor (quantized)
        torch::Tensor weight_ih = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight_hh tensor (quantized)
        torch::Tensor weight_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create bias_ih tensor
        torch::Tensor bias_ih = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create bias_hh tensor
        torch::Tensor bias_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create packed_ih tensor
        torch::Tensor packed_ih = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create packed_hh tensor
        torch::Tensor packed_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create col_offsets_ih tensor
        torch::Tensor col_offsets_ih = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create col_offsets_hh tensor
        torch::Tensor col_offsets_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse scale values for quantization
        double w_ih_scale = 1.0;
        double w_hh_scale = 1.0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&w_ih_scale, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&w_hh_scale, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse zero points for quantization
        int64_t w_ih_zero_point = 0;
        int64_t w_hh_zero_point = 0;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&w_ih_zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&w_hh_zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Try to apply the quantized_gru_cell operation
        torch::Tensor result;
        
        // Attempt to call quantized_gru_cell with the created tensors
        result = torch::quantized_gru_cell(
            input,
            hx,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            packed_ih,
            packed_hh,
            col_offsets_ih,
            col_offsets_hh,
            w_ih_scale,
            w_hh_scale,
            w_ih_zero_point,
            w_hh_zero_point
        );
        
        // Perform some operation on the result to ensure it's used
        if (result.defined()) {
            auto sum = result.sum();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}