#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) return 0;  // Need minimum data for meaningful test
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create hidden state tensors (h and c)
        torch::Tensor h_state = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor c_state = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensors
        torch::Tensor w_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor w_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create packed weight tensors
        torch::Tensor packed_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor packed_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create column offset tensors
        torch::Tensor col_offsets_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor col_offsets_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create bias tensors (optional)
        torch::Tensor b_ih, b_hh;
        if (offset < Size - 10) {
            b_ih = fuzzer_utils::createTensor(Data, Size, offset);
            b_hh = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Create scale parameters for quantization
        double scale_ih = 1.0;
        double scale_hh = 1.0;
        double zero_point_ih = 0;
        double zero_point_hh = 0;
        
        // Extract scale and zero point values from input data if available
        if (offset + 32 < Size) {
            std::memcpy(&scale_ih, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&scale_hh, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&zero_point_ih, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&zero_point_hh, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Try different variants of the quantized_lstm_cell call
        try {
            // Variant 1: With all parameters
            auto result1 = torch::quantized_lstm_cell(
                input, 
                {h_state, c_state},
                w_ih, w_hh, b_ih, b_hh,
                packed_ih, packed_hh,
                col_offsets_ih, col_offsets_hh,
                scale_ih, scale_hh, zero_point_ih, zero_point_hh
            );
        } catch (const std::exception& e) {
            // Continue to next variant
        }
        
        try {
            // Variant 2: Without bias
            auto result2 = torch::quantized_lstm_cell(
                input, 
                {h_state, c_state},
                w_ih, w_hh, 
                torch::Tensor(), torch::Tensor(),
                packed_ih, packed_hh,
                col_offsets_ih, col_offsets_hh,
                scale_ih, scale_hh, zero_point_ih, zero_point_hh
            );
        } catch (const std::exception& e) {
            // Continue to next variant
        }
        
        try {
            // Variant 3: With different scales and zero points
            auto result3 = torch::quantized_lstm_cell(
                input, 
                {h_state, c_state},
                w_ih, w_hh, b_ih, b_hh,
                packed_ih, packed_hh,
                col_offsets_ih, col_offsets_hh,
                0.01, 0.02, 10, 20
            );
        } catch (const std::exception& e) {
            // Continue
        }
        
        // Try with edge case values for scales and zero points
        try {
            auto result4 = torch::quantized_lstm_cell(
                input, 
                {h_state, c_state},
                w_ih, w_hh, b_ih, b_hh,
                packed_ih, packed_hh,
                col_offsets_ih, col_offsets_hh,
                std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 
                std::numeric_limits<double>::min(), std::numeric_limits<double>::max()
            );
        } catch (const std::exception& e) {
            // Continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}