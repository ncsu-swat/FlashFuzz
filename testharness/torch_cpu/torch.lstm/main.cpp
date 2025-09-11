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
        
        // Create hidden state tensors
        torch::Tensor h0 = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor c0 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensors
        torch::Tensor weight_ih = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor weight_hh = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create bias tensors (optional)
        torch::Tensor bias_ih, bias_hh;
        if (offset < Size - 2) {
            bias_ih = fuzzer_utils::createTensor(Data, Size, offset);
            bias_hh = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Get some configuration parameters from the remaining data
        bool has_biases = offset < Size && (Data[offset++] % 2 == 0);
        bool batch_first = offset < Size && (Data[offset++] % 2 == 0);
        bool bidirectional = offset < Size && (Data[offset++] % 2 == 0);
        int64_t num_layers = 1;
        if (offset < Size) {
            num_layers = (Data[offset++] % 3) + 1; // 1-3 layers
        }
        double dropout = 0.0;
        if (offset < Size) {
            dropout = static_cast<double>(Data[offset++]) / 255.0; // 0.0-1.0
        }
        
        // Try different variants of LSTM calls
        try {
            // Variant 1: Basic LSTM call with all required parameters
            at::TensorList hx_list = {h0, c0};
            at::TensorList params_list = {weight_ih, weight_hh};
            if (has_biases && bias_ih.defined() && bias_hh.defined()) {
                params_list = {weight_ih, weight_hh, bias_ih, bias_hh};
            }
            
            auto output1 = torch::lstm(input, hx_list, params_list, has_biases, 
                                     num_layers, dropout, true, bidirectional, batch_first);
            
            // Variant 2: LSTM with different parameter combinations
            if (h0.defined() && c0.defined() && weight_ih.defined() && weight_hh.defined()) {
                at::TensorList hx_list2 = {h0, c0};
                at::TensorList params_list2 = {weight_ih, weight_hh};
                
                auto output2 = torch::lstm(input, hx_list2, params_list2, false, 
                                         num_layers, dropout, true, bidirectional, batch_first);
            }
            
            // Variant 3: LSTM with packed sequence
            try {
                torch::Tensor lengths = torch::tensor({input.size(0)}, torch::kLong);
                auto packed_input = torch::nn::utils::rnn::pack_padded_sequence(input, lengths, batch_first);
                
                at::TensorList hx_list3 = {h0, c0};
                at::TensorList params_list3 = {weight_ih, weight_hh};
                
                auto output3 = torch::lstm(packed_input.data(), packed_input.batch_sizes(), 
                                         hx_list3, params_list3, has_biases, 
                                         num_layers, dropout, true, bidirectional);
            } catch (...) {
                // Packing might fail for invalid inputs, that's expected
            }
            
        } catch (const c10::Error& e) {
            // Expected PyTorch errors (e.g., shape mismatch) are fine
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
