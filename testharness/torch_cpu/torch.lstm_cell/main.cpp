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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create hidden state tensors (h0, c0)
        torch::Tensor h0 = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor c0 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensors
        torch::Tensor w_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor w_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create bias tensors (optional)
        bool use_bias = offset < Size && (Data[offset++] & 0x1);
        torch::Tensor b_ih, b_hh;
        
        if (use_bias) {
            b_ih = fuzzer_utils::createTensor(Data, Size, offset);
            b_hh = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Try different variants of lstm_cell
        try {
            // Basic variant with all parameters
            auto result1 = torch::lstm_cell(input, {h0, c0}, w_ih, w_hh, b_ih, b_hh);
            
            // Access the results to ensure they're used
            auto h_out = std::get<0>(result1);
            auto c_out = std::get<1>(result1);
            
            // Variant without bias
            if (!use_bias) {
                auto result2 = torch::lstm_cell(input, {h0, c0}, w_ih, w_hh);
                h_out = std::get<0>(result2);
                c_out = std::get<1>(result2);
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and normal during fuzzing
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
