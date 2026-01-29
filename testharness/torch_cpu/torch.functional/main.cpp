#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::max, std::swap

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test various torch::nn::functional operations
        if (offset + 1 < Size) {
            uint8_t op_selector = Data[offset++];
            
            // Select operation based on the selector byte
            switch (op_selector % 10) {
                case 0: {
                    // torch::nn::functional::relu
                    torch::Tensor result = torch::nn::functional::relu(input);
                    break;
                }
                case 1: {
                    // torch::nn::functional::gelu
                    torch::Tensor result = torch::nn::functional::gelu(input);
                    break;
                }
                case 2: {
                    // torch::nn::functional::softmax
                    if (input.dim() > 0 && offset < Size) {
                        int64_t dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input.dim());
                        try {
                            torch::Tensor result = torch::nn::functional::softmax(input, torch::nn::functional::SoftmaxFuncOptions(dim));
                        } catch (...) {
                            // Shape/dtype issues - ignore
                        }
                    }
                    break;
                }
                case 3: {
                    // torch::nn::functional::log_softmax
                    if (input.dim() > 0 && offset < Size) {
                        int64_t dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input.dim());
                        try {
                            torch::Tensor result = torch::nn::functional::log_softmax(input, torch::nn::functional::LogSoftmaxFuncOptions(dim));
                        } catch (...) {
                            // Shape/dtype issues - ignore
                        }
                    }
                    break;
                }
                case 4: {
                    // torch::nn::functional::dropout
                    if (offset < Size) {
                        double p = static_cast<double>(Data[offset++]) / 255.0;
                        bool train = (offset < Size) ? (Data[offset++] % 2 == 0) : true;
                        torch::Tensor result = torch::nn::functional::dropout(input, torch::nn::functional::DropoutFuncOptions().p(p).training(train));
                    }
                    break;
                }
                case 5: {
                    // torch::nn::functional::elu
                    if (offset < Size) {
                        double alpha = static_cast<double>(Data[offset++]) / 64.0;
                        torch::Tensor result = torch::nn::functional::elu(input, torch::nn::functional::ELUFuncOptions().alpha(alpha));
                    }
                    break;
                }
                case 6: {
                    // torch::nn::functional::selu
                    torch::Tensor result = torch::nn::functional::selu(input);
                    break;
                }
                case 7: {
                    // torch::hardsigmoid
                    torch::Tensor result = torch::hardsigmoid(input);
                    break;
                }
                case 8: {
                    // torch::hardtanh
                    if (offset + 1 < Size) {
                        double min_val = static_cast<double>(Data[offset++]) / 64.0 - 2.0;
                        double max_val = static_cast<double>(Data[offset++]) / 64.0 + 2.0;
                        if (min_val > max_val) {
                            std::swap(min_val, max_val);
                        }
                        torch::Tensor result = torch::hardtanh(input, min_val, max_val);
                    }
                    break;
                }
                case 9: {
                    // torch::nn::functional::leaky_relu
                    if (offset < Size) {
                        double negative_slope = static_cast<double>(Data[offset++]) / 128.0;
                        torch::Tensor result = torch::nn::functional::leaky_relu(input, torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope));
                    }
                    break;
                }
            }
        }
        
        // Create a second tensor with matching shape for binary operations
        if (offset + 3 < Size) {
            // Clone input and modify it to ensure compatible shapes
            torch::Tensor second_input = input.clone();
            
            // Add some variation based on fuzzer data
            if (offset < Size) {
                float scale = static_cast<float>(Data[offset++]) / 128.0f;
                second_input = second_input * scale;
            }
            
            // Test binary operations
            if (offset < Size) {
                uint8_t bin_op_selector = Data[offset++];
                
                switch (bin_op_selector % 5) {
                    case 0: {
                        // torch::nn::functional::mse_loss
                        try {
                            torch::Tensor result = torch::nn::functional::mse_loss(input, second_input);
                        } catch (...) {
                            // Shape mismatch - ignore
                        }
                        break;
                    }
                    case 1: {
                        // torch::nn::functional::binary_cross_entropy
                        try {
                            // Convert to float and clamp to [eps, 1-eps] for BCE
                            torch::Tensor float_input = input.to(torch::kFloat);
                            torch::Tensor float_target = second_input.to(torch::kFloat);
                            float_input = torch::clamp(torch::sigmoid(float_input), 1e-7, 1.0 - 1e-7);
                            float_target = torch::clamp(torch::sigmoid(float_target), 0.0, 1.0);
                            torch::Tensor result = torch::nn::functional::binary_cross_entropy(float_input, float_target);
                        } catch (...) {
                            // Expected failures - ignore
                        }
                        break;
                    }
                    case 2: {
                        // torch::nn::functional::cosine_similarity
                        if (input.dim() > 0 && offset < Size) {
                            int64_t dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input.dim());
                            try {
                                torch::Tensor result = torch::nn::functional::cosine_similarity(input, second_input, torch::nn::functional::CosineSimilarityFuncOptions().dim(dim));
                            } catch (...) {
                                // Shape mismatch - ignore
                            }
                        }
                        break;
                    }
                    case 3: {
                        // torch::nn::functional::pairwise_distance
                        try {
                            if (input.dim() >= 2 && offset < Size) {
                                double p = 2.0 + (static_cast<double>(Data[offset++]) / 64.0);
                                torch::Tensor result = torch::nn::functional::pairwise_distance(input, second_input, torch::nn::functional::PairwiseDistanceFuncOptions().p(p));
                            }
                        } catch (...) {
                            // Shape mismatch - ignore
                        }
                        break;
                    }
                    case 4: {
                        // torch::nn::functional::kl_div
                        try {
                            torch::Tensor result = torch::nn::functional::kl_div(input, second_input);
                        } catch (...) {
                            // Shape mismatch - ignore
                        }
                        break;
                    }
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}