#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least a few bytes for basic parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract in_features from fuzzer data
        int64_t in_features = 0;
        if (offset + sizeof(int32_t) <= Size) {
            int32_t tmp;
            std::memcpy(&tmp, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            // Make in_features reasonable (1 to 256)
            in_features = std::abs(tmp) % 256 + 1;
        } else {
            in_features = 16;
        }
        
        // Extract out_features from fuzzer data
        int64_t out_features = 0;
        if (offset + sizeof(int32_t) <= Size) {
            int32_t tmp;
            std::memcpy(&tmp, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            // Make out_features reasonable (1 to 256)
            out_features = std::abs(tmp) % 256 + 1;
        } else {
            out_features = 10;
        }
        
        // Get bias parameter if data available
        bool use_bias = true;
        if (offset < Size) {
            use_bias = Data[offset++] & 0x1;
        }
        
        // Create Linear module with both in_features and out_features specified
        auto linear = torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(use_bias));
        
        // Create input tensor with matching in_features dimension
        // Extract batch size from fuzzer data
        int64_t batch_size = 1;
        if (offset < Size) {
            batch_size = (Data[offset++] % 16) + 1;
        }
        
        // Create input tensor with correct shape [batch_size, in_features]
        torch::Tensor input = torch::randn({batch_size, in_features});
        
        // Apply Linear layer
        torch::Tensor output;
        try {
            output = linear(input);
            output = output.contiguous();
            
            // Force computation
            if (output.numel() > 0) {
                float sum = output.sum().item<float>();
                (void)sum;
            }
        } catch (const std::exception&) {
            // Shape mismatches are expected for some fuzzer inputs
        }
        
        // Test with different batch sizes
        try {
            int64_t batch2 = (Size > offset && offset < Size) ? (Data[offset % Size] % 32 + 1) : 8;
            torch::Tensor input2 = torch::randn({batch2, in_features});
            torch::Tensor output2 = linear(input2);
            output2 = output2.contiguous();
            
            if (output2.numel() > 0) {
                float mean = output2.mean().item<float>();
                (void)mean;
            }
        } catch (const std::exception&) {
            // Ignore errors
        }
        
        // Test with 3D input (batch, seq, features)
        try {
            int64_t seq_len = (Size > offset && offset < Size) ? (Data[offset % Size] % 8 + 1) : 3;
            torch::Tensor input3d = torch::randn({2, seq_len, in_features});
            torch::Tensor output3d = linear(input3d);
            output3d = output3d.contiguous();
            
            if (output3d.numel() > 0) {
                float val = output3d.sum().item<float>();
                (void)val;
            }
        } catch (const std::exception&) {
            // Ignore errors
        }
        
        // Test zero batch size
        try {
            torch::Tensor zero_batch = torch::empty({0, in_features});
            torch::Tensor zero_output = linear(zero_batch);
            zero_output = zero_output.contiguous();
        } catch (const std::exception&) {
            // Ignore zero-batch errors
        }
        
        // Test accessing parameters
        try {
            if (linear->weight.defined()) {
                auto weight_shape = linear->weight.sizes();
                (void)weight_shape;
                
                // Verify weight shape
                if (weight_shape.size() == 2) {
                    float w_sum = linear->weight.sum().item<float>();
                    (void)w_sum;
                }
            }
            if (use_bias && linear->bias.defined()) {
                auto bias_shape = linear->bias.sizes();
                (void)bias_shape;
                
                float b_sum = linear->bias.sum().item<float>();
                (void)b_sum;
            }
        } catch (const std::exception&) {
            // Ignore errors
        }
        
        // Test with tensor created from fuzzer data
        try {
            torch::Tensor fuzz_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape to match expected input dimensions if possible
            if (fuzz_input.numel() >= in_features) {
                int64_t fuzz_batch = fuzz_input.numel() / in_features;
                if (fuzz_batch > 0) {
                    fuzz_input = fuzz_input.reshape({fuzz_batch, in_features});
                    if (!fuzz_input.is_floating_point()) {
                        fuzz_input = fuzz_input.to(torch::kFloat32);
                    }
                    torch::Tensor fuzz_output = linear(fuzz_input);
                    fuzz_output = fuzz_output.contiguous();
                }
            }
        } catch (const std::exception&) {
            // Ignore shape mismatch errors
        }
        
        // Test with a fresh Linear module with different parameters
        try {
            int64_t in2 = (in_features % 64) + 1;
            int64_t out2 = (out_features % 64) + 1;
            auto linear2 = torch::nn::Linear(torch::nn::LinearOptions(in2, out2).bias(!use_bias));
            
            torch::Tensor test_input = torch::randn({4, in2});
            torch::Tensor test_output = linear2(test_input);
            test_output = test_output.contiguous();
            
            if (test_output.numel() > 0) {
                float val = test_output.mean().item<float>();
                (void)val;
            }
        } catch (const std::exception&) {
            // Ignore errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}