#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) return 0;
        
        // Create tensors for distribution parameters
        torch::Tensor loc = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data left
        if (offset >= Size) return 0;
        
        torch::Tensor scale = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to create and use various distributions
        if (offset < Size) {
            uint8_t dist_type = Data[offset++] % 8;
            
            try {
                if (dist_type == 0 && loc.dim() == 0 && scale.dim() == 0) {
                    // Normal distribution
                    auto normal = torch::normal(loc, scale.abs());
                }
                else if (dist_type == 1 && loc.dim() == 0 && scale.dim() == 0) {
                    // Bernoulli distribution
                    auto probs = torch::sigmoid(loc);
                    auto bernoulli = torch::bernoulli(probs);
                }
                else if (dist_type == 2 && loc.dim() == 0 && scale.dim() == 0) {
                    // Exponential distribution
                    auto rate = scale.abs() + 1e-5;  // Ensure rate is positive
                    auto exponential = torch::exponential(rate);
                }
                else if (dist_type == 3) {
                    // Categorical distribution
                    auto probs = torch::softmax(loc, -1);
                    auto categorical = torch::multinomial(probs, 1);
                }
                else if (dist_type == 4 && loc.dim() == 0 && scale.dim() == 0) {
                    // Uniform distribution
                    auto low = loc;
                    auto high = loc + scale.abs() + 1e-5;  // Ensure high > low
                    auto uniform = torch::rand_like(loc) * (high - low) + low;
                }
                else if (dist_type == 5 && loc.dim() == 0 && scale.dim() == 0) {
                    // Gamma distribution
                    auto concentration = scale.abs() + 1e-5;  // Ensure positive
                    auto rate = loc.abs() + 1e-5;  // Ensure positive
                    auto gamma = torch::gamma(concentration);
                }
                else if (dist_type == 6 && loc.dim() > 0) {
                    // Multinomial distribution
                    auto probs = torch::softmax(loc, -1);
                    int64_t total_count = 10;
                    auto multinomial = torch::multinomial(probs, total_count, true);
                }
                else if (dist_type == 7 && loc.dim() == 0 && scale.dim() == 0) {
                    // Poisson distribution
                    auto rate = scale.abs() + 1e-5;  // Ensure rate is positive
                    auto poisson = torch::poisson(rate);
                }
            } catch (const std::exception &e) {
                // Distribution-specific exceptions are expected and handled here
            }
        }
        
        // Try to create a multivariate normal if we have enough data
        if (offset < Size) {
            try {
                // Create a covariance matrix
                int64_t dim = 2 + (Data[offset++] % 3);  // 2-4 dimensions
                auto loc_mv = torch::randn({dim});
                
                // Create a positive definite covariance matrix
                auto cov_raw = torch::randn({dim, dim});
                auto cov = torch::matmul(cov_raw, cov_raw.transpose(0, 1));
                cov.add_(torch::eye(dim).mul(0.01));  // Ensure positive definiteness
                
                auto mvn = torch::mv(cov, loc_mv);
            } catch (const std::exception &e) {
                // MultivariateNormal-specific exceptions are expected and handled here
            }
        }
        
        // Try to create a mixture distribution if we have enough data
        if (offset < Size) {
            try {
                int64_t num_components = 2 + (Data[offset++] % 3);  // 2-4 components
                std::vector<torch::Tensor> components;
                
                for (int64_t i = 0; i < num_components; i++) {
                    auto comp_loc = torch::randn({});
                    auto comp_scale = torch::rand({}) + 0.1;  // Ensure positive scale
                    components.push_back(torch::normal(comp_loc, comp_scale));
                }
                
                auto mix_probs = torch::softmax(torch::randn({num_components}), 0);
                auto mixture = torch::multinomial(mix_probs, 1);
                
                // Sample from the mixture
                auto component_idx = mixture.item<int64_t>();
                if (component_idx < components.size()) {
                    auto sample = components[component_idx];
                }
            } catch (const std::exception &e) {
                // Mixture-specific exceptions are expected and handled here
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