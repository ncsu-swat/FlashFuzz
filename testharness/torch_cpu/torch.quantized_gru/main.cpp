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
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse parameters for GRU
        uint8_t has_bias = (offset < Size) ? Data[offset++] & 0x1 : 0;
        uint8_t num_layers = (offset < Size) ? (Data[offset++] % 3) + 1 : 1;
        uint8_t dropout_byte = (offset < Size) ? Data[offset++] : 0;
        float dropout = dropout_byte / 255.0f;
        uint8_t bidirectional = (offset < Size) ? Data[offset++] & 0x1 : 0;
        uint8_t batch_first = (offset < Size) ? Data[offset++] & 0x1 : 0;
        
        // Create weight and bias tensors
        int64_t input_size = input.size(input.dim() > 2 && batch_first ? 2 : 1);
        int64_t hidden_size = (offset < Size) ? (Data[offset++] % 32) + 1 : 8;
        
        // Create weight_ih and weight_hh
        torch::Tensor weight_ih, weight_hh, bias_ih, bias_hh;
        
        try {
            // Create weight tensors
            std::vector<int64_t> weight_ih_shape = {3 * hidden_size, input_size};
            std::vector<int64_t> weight_hh_shape = {3 * hidden_size, hidden_size};
            
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            weight_ih = torch::randn(weight_ih_shape, options);
            weight_hh = torch::randn(weight_hh_shape, options);
            
            // Create bias tensors if needed
            if (has_bias) {
                std::vector<int64_t> bias_shape = {3 * hidden_size};
                bias_ih = torch::randn(bias_shape, options);
                bias_hh = torch::randn(bias_shape, options);
            }
            
            // Create scale and zero_point tensors
            float scale_value = 0.1f;
            int64_t zero_point = 0;
            
            // Quantize the tensors
            weight_ih = torch::quantize_per_tensor(weight_ih, scale_value, zero_point, torch::kQInt8);
            weight_hh = torch::quantize_per_tensor(weight_hh, scale_value, zero_point, torch::kQInt8);
            
            if (has_bias) {
                bias_ih = torch::quantize_per_tensor(bias_ih, scale_value, zero_point, torch::kQInt32);
                bias_hh = torch::quantize_per_tensor(bias_hh, scale_value, zero_point, torch::kQInt32);
            }
            
            // Create packed weights using torch::ops::quantized::gru_cell_packed_weight
            std::vector<torch::Tensor> weights;
            weights.push_back(weight_ih);
            weights.push_back(weight_hh);
            if (has_bias) {
                weights.push_back(bias_ih);
                weights.push_back(bias_hh);
            }
            
            // Create initial hidden state
            std::vector<int64_t> h0_shape = {num_layers * (bidirectional ? 2 : 1), 
                                            input.size(batch_first ? 0 : 1), 
                                            hidden_size};
            torch::Tensor h0 = torch::zeros(h0_shape);
            
            // Apply quantized_gru using torch::ops::quantized::gru
            auto gru = torch::ops::quantized::gru(
                input.to(torch::kFloat),
                h0,
                weights,
                has_bias,
                num_layers,
                dropout,
                bidirectional == 1,
                batch_first == 1
            );
            
            // Access the output to ensure computation is not optimized away
            auto output = std::get<0>(gru);
            auto h_n = std::get<1>(gru);
            
            // Force evaluation
            output.sum().item<float>();
            h_n.sum().item<float>();
        }
        catch (const c10::Error& e) {
            // PyTorch specific errors are expected and not a fuzzer issue
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
