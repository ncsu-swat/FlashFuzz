#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        
        // Create input tensors
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a byte to determine which distance module to test
        uint8_t module_selector = 0;
        if (offset < Size) {
            module_selector = Data[offset++];
        }
        
        // Get a byte for parameter configuration
        uint8_t param_byte = 0;
        if (offset < Size) {
            param_byte = Data[offset++];
        }
        
        // Test PairwiseDistance module
        if (module_selector % 3 == 0) {
            double p = 2.0;
            if (param_byte % 3 == 0) p = 0.5;
            else if (param_byte % 3 == 1) p = 1.0;
            
            bool keepdim = (param_byte & 0x10) != 0;
            double eps = 1e-6;
            
            auto pairwise_distance = torch::nn::PairwiseDistance(
                torch::nn::PairwiseDistanceOptions()
                    .p(p)
                    .eps(eps)
                    .keepdim(keepdim)
            );
            
            try {
                auto output = pairwise_distance->forward(x1, x2);
            } catch (const c10::Error &e) {
                // Shape mismatch or other expected errors
            }
        }
        // Test CosineSimilarity module
        else if (module_selector % 3 == 1) {
            int64_t dim = param_byte % 4;
            if (x1.dim() > 0 && dim >= x1.dim()) {
                dim = x1.dim() - 1;
            }
            if (dim < 0) {
                dim = 0;
            }
            
            double eps = 1e-8;
            
            auto cosine_similarity = torch::nn::CosineSimilarity(
                torch::nn::CosineSimilarityOptions()
                    .dim(dim)
                    .eps(eps)
            );
            
            try {
                auto output = cosine_similarity->forward(x1, x2);
            } catch (const c10::Error &e) {
                // Shape mismatch or other expected errors
            }
        }
        // Test CosineEmbeddingLoss module
        else {
            double margin = ((param_byte % 21) - 10) / 10.0;
            
            int64_t reduction = param_byte % 3;
            torch::nn::CosineEmbeddingLossOptions::reduction_t reduction_enum;
            
            if (reduction == 0) {
                reduction_enum = torch::kNone;
            } else if (reduction == 1) {
                reduction_enum = torch::kMean;
            } else {
                reduction_enum = torch::kSum;
            }
            
            auto cosine_embedding_loss = torch::nn::CosineEmbeddingLoss(
                torch::nn::CosineEmbeddingLossOptions()
                    .margin(margin)
                    .reduction(reduction_enum)
            );
            
            // Create target tensor with values -1 or 1 (must be float type)
            std::vector<int64_t> target_shape;
            if (x1.dim() > 0 && x2.dim() > 0) {
                for (int i = 0; i < x1.dim() - 1; i++) {
                    if (i < static_cast<int>(x1.sizes().size())) {
                        target_shape.push_back(x1.size(i));
                    }
                }
            }
            
            if (target_shape.empty()) {
                target_shape.push_back(1);
            }
            
            torch::Tensor target;
            if (offset < Size) {
                // Use remaining data to create target values
                std::vector<float> target_values;
                int64_t num_elements = 1;
                for (auto d : target_shape) {
                    num_elements *= d;
                }
                
                for (int64_t i = 0; i < num_elements && offset < Size; i++) {
                    target_values.push_back(Data[offset++] % 2 == 0 ? 1.0f : -1.0f);
                }
                
                if (!target_values.empty()) {
                    target = torch::tensor(target_values).reshape(target_shape);
                } else {
                    target = torch::ones(target_shape);
                }
            } else {
                target = torch::ones(target_shape);
            }
            
            try {
                auto output = cosine_embedding_loss->forward(x1, x2, target);
            } catch (const c10::Error &e) {
                // Shape mismatch or other expected errors
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