#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
                        torch::Tensor result = torch::nn::functional::softmax(input, torch::nn::functional::SoftmaxFuncOptions(dim));
                    }
                    break;
                }
                case 3: {
                    // torch::nn::functional::log_softmax
                    if (input.dim() > 0 && offset < Size) {
                        int64_t dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input.dim());
                        torch::Tensor result = torch::nn::functional::log_softmax(input, torch::nn::functional::LogSoftmaxFuncOptions(dim));
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
        
        // Create a second tensor if there's enough data left
        if (offset + 3 < Size) {
            torch::Tensor second_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test binary operations
            if (offset < Size) {
                uint8_t bin_op_selector = Data[offset++];
                
                switch (bin_op_selector % 5) {
                    case 0: {
                        // torch::nn::functional::mse_loss
                        torch::Tensor result = torch::nn::functional::mse_loss(input, second_input);
                        break;
                    }
                    case 1: {
                        // torch::nn::functional::binary_cross_entropy
                        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                            // Clamp input and target to [0, 1] for BCE
                            torch::Tensor clamped_input = torch::clamp(input, 0.0, 1.0);
                            torch::Tensor clamped_target = torch::clamp(second_input, 0.0, 1.0);
                            torch::Tensor result = torch::nn::functional::binary_cross_entropy(clamped_input, clamped_target);
                        }
                        break;
                    }
                    case 2: {
                        // torch::nn::functional::cosine_similarity
                        if (input.dim() > 0 && offset < Size) {
                            int64_t dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input.dim());
                            torch::Tensor result = torch::nn::functional::cosine_similarity(input, second_input, torch::nn::functional::CosineSimilarityFuncOptions().dim(dim));
                        }
                        break;
                    }
                    case 3: {
                        // torch::nn::functional::pairwise_distance
                        if (input.dim() > 0 && offset < Size) {
                            double p = 2.0 + (static_cast<double>(Data[offset++]) / 64.0);
                            torch::Tensor result = torch::nn::functional::pairwise_distance(input, second_input, torch::nn::functional::PairwiseDistanceFuncOptions().p(p));
                        }
                        break;
                    }
                    case 4: {
                        // torch::nn::functional::kl_div
                        torch::Tensor result = torch::nn::functional::kl_div(input, second_input);
                        break;
                    }
                }
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
