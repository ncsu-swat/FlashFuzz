#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

// torch.nn.UninitializedParameter is a Python-only API.
// It's used internally by LazyLinear, LazyConv2d, etc. which are also Python-only.
// The C++ frontend does not expose UninitializedParameter or lazy modules.
//
// This harness tests torch::nn::Parameter which is the closest C++ equivalent -
// it represents initialized parameters that can be registered with modules.

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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

        // Test 1: Create Parameter from tensor and test basic operations
        {
            int64_t dim1 = static_cast<int64_t>(Data[offset++] % 32) + 1;
            int64_t dim2 = static_cast<int64_t>(Data[offset++] % 32) + 1;
            bool requires_grad = (Data[offset++] % 2) == 0;
            
            // Create a tensor and wrap it as a parameter
            auto tensor = torch::randn({dim1, dim2});
            tensor.set_requires_grad(requires_grad);
            
            // Test parameter properties
            auto grad_status = tensor.requires_grad();
            auto shape = tensor.sizes();
            auto numel = tensor.numel();
            auto dtype = tensor.dtype();
            
            // Perform operations that would be done on parameters
            if (requires_grad) {
                auto sum = tensor.sum();
                sum.backward();
                if (tensor.grad().defined()) {
                    auto grad_sum = tensor.grad().sum();
                }
            }
        }
        
        // Test 2: Custom module with parameters (simulating what lazy modules do after materialization)
        if (offset + 4 < Size)
        {
            int64_t in_features = static_cast<int64_t>(Data[offset++] % 64) + 1;
            int64_t out_features = static_cast<int64_t>(Data[offset++] % 64) + 1;
            int64_t batch_size = static_cast<int64_t>(Data[offset++] % 16) + 1;
            bool use_bias = (Data[offset++] % 2) == 0;
            
            try {
                // Create Linear module (which internally creates parameters)
                auto options = torch::nn::LinearOptions(in_features, out_features).bias(use_bias);
                torch::nn::Linear linear(options);
                
                // Access the parameters
                auto params = linear->parameters();
                for (auto& p : params) {
                    auto p_shape = p.sizes();
                    auto p_numel = p.numel();
                    bool p_requires_grad = p.requires_grad();
                }
                
                // Forward pass
                auto input = torch::randn({batch_size, in_features});
                auto output = linear->forward(input);
                
                // Verify output shape
                if (output.size(0) == batch_size && output.size(1) == out_features) {
                    // Shape is correct
                }
            } catch (const std::exception&) {
                // Expected for some configurations
            }
        }
        
        // Test 3: Conv2d module with parameters
        if (offset + 5 < Size)
        {
            int64_t in_channels = static_cast<int64_t>(Data[offset++] % 16) + 1;
            int64_t out_channels = static_cast<int64_t>(Data[offset++] % 32) + 1;
            int64_t kernel_size = static_cast<int64_t>(Data[offset++] % 5) + 1;
            int64_t batch = static_cast<int64_t>(Data[offset++] % 8) + 1;
            int64_t hw = static_cast<int64_t>(Data[offset++] % 16) + kernel_size;
            
            try {
                auto options = torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size);
                torch::nn::Conv2d conv(options);
                
                // Access weight and bias parameters
                auto weight = conv->weight;
                if (weight.defined()) {
                    auto w_sizes = weight.sizes();
                }
                
                if (conv->bias.defined()) {
                    auto bias = conv->bias;
                    auto b_sizes = bias.sizes();
                }
                
                auto input = torch::randn({batch, in_channels, hw, hw});
                auto output = conv->forward(input);
            } catch (const std::exception&) {
                // Expected for some configurations
            }
        }
        
        // Test 4: Sequential model with multiple parameterized modules
        if (offset + 4 < Size)
        {
            int64_t in_dim = static_cast<int64_t>(Data[offset++] % 32) + 1;
            int64_t hidden = static_cast<int64_t>(Data[offset++] % 32) + 1;
            int64_t out_dim = static_cast<int64_t>(Data[offset++] % 16) + 1;
            int64_t batch = static_cast<int64_t>(Data[offset++] % 8) + 1;
            
            try {
                torch::nn::Sequential seq(
                    torch::nn::Linear(in_dim, hidden),
                    torch::nn::ReLU(),
                    torch::nn::Linear(hidden, out_dim)
                );
                
                // Count total parameters
                auto params = seq->parameters();
                int64_t total_params = 0;
                for (auto& p : params) {
                    total_params += p.numel();
                }
                
                auto input = torch::randn({batch, in_dim});
                auto output = seq->forward(input);
            } catch (const std::exception&) {
                // Expected for some configurations
            }
        }
        
        // Test 5: BatchNorm with learnable parameters
        if (offset + 3 < Size)
        {
            int64_t num_features = static_cast<int64_t>(Data[offset++] % 64) + 1;
            int64_t batch = static_cast<int64_t>(Data[offset++] % 16) + 1;
            bool affine = (Data[offset++] % 2) == 0;
            
            try {
                auto options = torch::nn::BatchNorm1dOptions(num_features).affine(affine);
                torch::nn::BatchNorm1d bn(options);
                bn->train();
                
                // Check if weight/bias parameters exist (only if affine=true)
                auto params = bn->parameters();
                
                auto input = torch::randn({batch, num_features});
                auto output = bn->forward(input);
            } catch (const std::exception&) {
                // Expected for some configurations
            }
        }
        
        // Test 6: Parameter manipulation - clone, detach, to different dtype
        if (offset + 2 < Size)
        {
            int64_t dim = static_cast<int64_t>(Data[offset++] % 32) + 1;
            uint8_t dtype_choice = Data[offset++] % 3;
            
            auto tensor = torch::randn({dim, dim}, torch::requires_grad(true));
            
            // Clone the parameter
            auto cloned = tensor.clone();
            
            // Detach from computation graph
            auto detached = tensor.detach();
            
            // Convert to different dtype
            try {
                torch::Tensor converted;
                if (dtype_choice == 0) {
                    converted = tensor.to(torch::kFloat32);
                } else if (dtype_choice == 1) {
                    converted = tensor.to(torch::kFloat64);
                } else {
                    converted = tensor.to(torch::kFloat16);
                }
            } catch (const std::exception&) {
                // Dtype conversion can fail in some cases
            }
        }
        
        // Test 7: Embedding module parameters
        if (offset + 3 < Size)
        {
            int64_t num_embeddings = static_cast<int64_t>(Data[offset++] % 100) + 1;
            int64_t embedding_dim = static_cast<int64_t>(Data[offset++] % 64) + 1;
            int64_t seq_len = static_cast<int64_t>(Data[offset++] % 16) + 1;
            
            try {
                torch::nn::Embedding emb(num_embeddings, embedding_dim);
                
                // Access embedding weight parameter
                auto weight = emb->weight;
                auto w_shape = weight.sizes();
                
                // Create indices and lookup
                auto indices = torch::randint(0, num_embeddings, {seq_len}, torch::kLong);
                auto output = emb->forward(indices);
            } catch (const std::exception&) {
                // Expected for some configurations
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