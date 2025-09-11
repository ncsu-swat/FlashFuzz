#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test various quantization operations
        
        // 1. Test basic quantization operations
        if (offset + 1 < Size) {
            uint8_t qtype = Data[offset++];
            
            // Try different quantization operations
            try {
                // Quantize the tensor
                auto scale = 1.0f / 256.0f;
                auto zero_point = 0;
                
                // Create quantized tensor
                auto quantized = torch::quantize_per_tensor(
                    input_tensor.to(torch::kFloat), 
                    scale, 
                    zero_point, 
                    torch::kQUInt8);
                
                // Test dequantize
                auto dequantized = torch::dequantize(quantized);
                
                // Test quantized operations if we have a valid quantized tensor
                if (qtype % 4 == 0 && input_tensor.dim() >= 2) {
                    // Test quantized linear using torch::nn::functional
                    auto weight = torch::randn({input_tensor.size(input_tensor.dim()-1), 10});
                    auto weight_scale = 1.0f / 128.0f;
                    auto weight_zero_point = 0;
                    auto quantized_weight = torch::quantize_per_tensor(
                        weight, 
                        weight_scale, 
                        weight_zero_point, 
                        torch::kQUInt8);
                    
                    auto bias = torch::randn({10});
                    
                    // Test basic linear operation with dequantized tensors
                    auto result = torch::nn::functional::linear(
                        dequantized, 
                        weight, 
                        bias);
                }
            } catch (const std::exception& e) {
                // Continue with other tests
            }
        }
        
        // 2. Test dynamic quantization simulation
        if (offset + 1 < Size) {
            uint8_t qconfig_type = Data[offset++];
            
            try {
                // Create a simple model to test
                torch::nn::Sequential model(
                    torch::nn::Linear(10, 10),
                    torch::nn::ReLU()
                );
                
                // Prepare the model for testing
                model->eval();
                
                // If we have input of right shape, test the model
                if (input_tensor.dim() > 0 && input_tensor.size(0) > 0) {
                    try {
                        // Reshape input to match model expectations if needed
                        auto reshaped_input = input_tensor;
                        if (input_tensor.dim() == 1 && input_tensor.size(0) >= 10) {
                            reshaped_input = input_tensor.slice(0, 0, 10).reshape({1, 10});
                        } else if (input_tensor.dim() >= 2) {
                            // Try to use the tensor as a batch
                            auto last_dim = input_tensor.size(-1);
                            if (last_dim >= 10) {
                                reshaped_input = input_tensor.slice(-1, 0, 10);
                            }
                        }
                        
                        // Convert to float for model input
                        reshaped_input = reshaped_input.to(torch::kFloat);
                        
                        // Run the model
                        auto output = model->forward(reshaped_input);
                        
                        // Simulate quantization by quantizing the output
                        auto quantized_output = torch::quantize_per_tensor(
                            output, 
                            1.0f / 128.0f, 
                            0, 
                            torch::kQUInt8);
                    } catch (const std::exception& e) {
                        // Continue with other tests
                    }
                }
            } catch (const std::exception& e) {
                // Continue with other tests
            }
        }
        
        // 3. Test pruning simulation (manual sparsity)
        if (offset + 1 < Size) {
            uint8_t ns_type = Data[offset++];
            
            try {
                // Test pruning functionality simulation
                if (ns_type % 3 == 0) {
                    // Create a simple model
                    torch::nn::Linear linear(10, 10);
                    
                    // Simulate pruning by zeroing out weights
                    float sparsity = 0.5;
                    auto weight = linear->weight.clone();
                    auto mask = torch::rand_like(weight) > sparsity;
                    weight = weight * mask.to(weight.dtype());
                    linear->weight.data().copy_(weight);
                    
                    // Test the pruned model if we have appropriate input
                    if (input_tensor.dim() > 0) {
                        try {
                            auto reshaped_input = input_tensor;
                            if (input_tensor.dim() == 1 && input_tensor.size(0) >= 10) {
                                reshaped_input = input_tensor.slice(0, 0, 10).reshape({1, 10});
                            }
                            
                            // Convert to float for model input
                            reshaped_input = reshaped_input.to(torch::kFloat);
                            
                            // Forward pass through pruned layer
                            auto output = linear->forward(reshaped_input);
                        } catch (const std::exception& e) {
                            // Continue with other tests
                        }
                    }
                }
            } catch (const std::exception& e) {
                // Continue with other tests
            }
        }
        
        // 4. Test sparsity simulation
        if (offset + 1 < Size) {
            uint8_t sparsity_type = Data[offset++];
            
            try {
                // Test sparsity functionality simulation
                torch::nn::Linear linear(10, 10);
                
                // Apply different sparsity configurations
                float sparsity = static_cast<float>(sparsity_type % 100) / 100.0f;
                
                // Simulate sparsity by manually zeroing weights
                auto weight = linear->weight.clone();
                auto mask = torch::rand_like(weight) > sparsity;
                weight = weight * mask.to(weight.dtype());
                linear->weight.data().copy_(weight);
                
                // Test the sparse model if we have appropriate input
                if (input_tensor.dim() > 0) {
                    try {
                        auto reshaped_input = input_tensor;
                        if (input_tensor.dim() == 1 && input_tensor.size(0) >= 10) {
                            reshaped_input = input_tensor.slice(0, 0, 10).reshape({1, 10});
                        }
                        
                        // Convert to float for model input
                        reshaped_input = reshaped_input.to(torch::kFloat);
                        
                        // Forward pass through sparse layer
                        auto output = linear->forward(reshaped_input);
                    } catch (const std::exception& e) {
                        // Continue with other tests
                    }
                }
            } catch (const std::exception& e) {
                // Continue with other tests
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
