#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset + 1 >= Size) {
            return 0;
        }
        
        uint8_t op_selector = Data[offset++];
        
        switch (op_selector % 12) {
            case 0: {
                // Test relu
                auto output = torch::relu(input);
                break;
            }
            case 1: {
                // Test sigmoid
                auto output = torch::sigmoid(input);
                break;
            }
            case 2: {
                // Test tanh
                auto output = torch::tanh(input);
                break;
            }
            case 3: {
                // Test softmax - need valid dimension
                try {
                    int64_t dim = input.dim() > 0 ? (static_cast<int64_t>(Data[offset % Size]) % input.dim()) : 0;
                    offset++;
                    auto output = torch::softmax(input, dim);
                } catch (...) {
                    // Shape mismatch, silently ignore
                }
                break;
            }
            case 4: {
                // Test dropout
                if (offset + sizeof(float) >= Size) break;
                float p = 0.0f;
                std::memcpy(&p, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Clamp p to valid range [0, 1)
                p = std::fmod(std::abs(p), 1.0f);
                if (std::isnan(p) || std::isinf(p)) p = 0.5f;
                auto output = torch::dropout(input, p, /*train=*/true);
                break;
            }
            case 5: {
                // Test log_softmax - need valid dimension
                try {
                    int64_t dim = input.dim() > 0 ? (static_cast<int64_t>(Data[offset % Size]) % input.dim()) : 0;
                    offset++;
                    auto output = torch::log_softmax(input, dim);
                } catch (...) {
                    // Shape mismatch, silently ignore
                }
                break;
            }
            case 6: {
                // Test leaky_relu
                if (offset + sizeof(float) >= Size) break;
                float negative_slope = 0.0f;
                std::memcpy(&negative_slope, Data + offset, sizeof(float));
                offset += sizeof(float);
                negative_slope = std::fmod(std::abs(negative_slope), 1.0f);
                if (std::isnan(negative_slope) || std::isinf(negative_slope)) negative_slope = 0.01f;
                auto output = torch::leaky_relu(input, negative_slope);
                break;
            }
            case 7: {
                // Test gelu
                auto output = torch::gelu(input);
                break;
            }
            case 8: {
                // Test elu
                auto output = torch::elu(input);
                break;
            }
            case 9: {
                // Test selu
                auto output = torch::selu(input);
                break;
            }
            case 10: {
                // Test hardtanh
                try {
                    float min_val = -1.0f;
                    float max_val = 1.0f;
                    if (offset + 2 * sizeof(float) <= Size) {
                        std::memcpy(&min_val, Data + offset, sizeof(float));
                        offset += sizeof(float);
                        std::memcpy(&max_val, Data + offset, sizeof(float));
                        offset += sizeof(float);
                        // Ensure valid range
                        if (std::isnan(min_val) || std::isinf(min_val)) min_val = -1.0f;
                        if (std::isnan(max_val) || std::isinf(max_val)) max_val = 1.0f;
                        if (min_val > max_val) std::swap(min_val, max_val);
                    }
                    auto output = torch::hardtanh(input, min_val, max_val);
                } catch (...) {
                    // Invalid params, silently ignore
                }
                break;
            }
            case 11: {
                // Test threshold
                try {
                    float threshold = 0.0f;
                    float value = 0.0f;
                    if (offset + 2 * sizeof(float) <= Size) {
                        std::memcpy(&threshold, Data + offset, sizeof(float));
                        offset += sizeof(float);
                        std::memcpy(&value, Data + offset, sizeof(float));
                        offset += sizeof(float);
                        if (std::isnan(threshold) || std::isinf(threshold)) threshold = 0.0f;
                        if (std::isnan(value) || std::isinf(value)) value = 0.0f;
                    }
                    auto output = torch::threshold(input, threshold, value);
                } catch (...) {
                    // Invalid params, silently ignore
                }
                break;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}