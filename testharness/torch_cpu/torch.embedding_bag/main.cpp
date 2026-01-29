#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with sort result

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
        // Need at least some data to proceed
        if (Size < 16) {
            return 0;
        }

        // Reserve first 8 bytes for parameters, use rest for tensor data
        size_t param_offset = 0;
        
        // Get mode parameter (0=sum, 1=mean, 2=max)
        int64_t mode = static_cast<int64_t>(Data[param_offset++]) % 3;
        
        // Get sparse parameter
        bool sparse = static_cast<bool>(Data[param_offset++] & 1);
        
        // Get scale_grad_by_freq parameter
        bool scale_grad_by_freq = static_cast<bool>(Data[param_offset++] & 1);
        
        // Get include_last_offset parameter
        bool include_last_offset = static_cast<bool>(Data[param_offset++] & 1);
        
        // Get padding_idx selector
        uint8_t padding_selector = Data[param_offset++];
        
        // Get per_sample_weights selector
        bool use_per_sample_weights = static_cast<bool>(Data[param_offset++] & 1);
        // per_sample_weights is only valid for sum and mean modes, not max
        if (mode == 2) {
            use_per_sample_weights = false;
        }
        
        // Get num_embeddings and embedding_dim from data
        int64_t num_embeddings = static_cast<int64_t>(Data[param_offset++] % 64) + 1;  // 1-64
        int64_t embedding_dim = static_cast<int64_t>(Data[param_offset++] % 32) + 1;   // 1-32
        
        size_t tensor_offset = param_offset;
        
        // Create weight tensor (embedding table) with controlled dimensions
        torch::Tensor weight = torch::randn({num_embeddings, embedding_dim}, torch::kFloat32);
        
        // Fill weight with fuzzed data if available
        if (tensor_offset + 4 <= Size) {
            size_t weight_data_size = std::min(Size - tensor_offset, 
                static_cast<size_t>(num_embeddings * embedding_dim * sizeof(float)));
            if (weight_data_size >= 4) {
                auto weight_data = weight.data_ptr<float>();
                size_t num_floats = weight_data_size / sizeof(float);
                for (size_t i = 0; i < num_floats && i < static_cast<size_t>(weight.numel()); i++) {
                    float val;
                    memcpy(&val, Data + tensor_offset + i * sizeof(float), sizeof(float));
                    // Sanitize the value to avoid NaN/Inf issues
                    if (std::isnan(val) || std::isinf(val)) {
                        val = 0.0f;
                    }
                    weight_data[i] = val;
                }
                tensor_offset += weight_data_size;
            }
        }
        
        // Create indices tensor from remaining data
        int64_t num_indices = 0;
        if (tensor_offset < Size) {
            num_indices = static_cast<int64_t>((Size - tensor_offset) % 64) + 1;  // 1-64 indices
        } else {
            num_indices = 8;  // Default
        }
        
        std::vector<int64_t> indices_vec;
        indices_vec.reserve(num_indices);
        for (int64_t i = 0; i < num_indices; i++) {
            int64_t idx;
            if (tensor_offset < Size) {
                idx = static_cast<int64_t>(Data[tensor_offset++]) % num_embeddings;
            } else {
                idx = i % num_embeddings;
            }
            indices_vec.push_back(idx);
        }
        torch::Tensor indices = torch::tensor(indices_vec, torch::kInt64);
        
        // Create offsets tensor
        int64_t num_bags = 0;
        if (tensor_offset < Size) {
            num_bags = static_cast<int64_t>(Data[tensor_offset++] % 8) + 1;  // 1-8 bags
        } else {
            num_bags = 2;  // Default
        }
        
        // Generate valid offsets (must be sorted and within [0, num_indices])
        std::vector<int64_t> offsets_vec;
        offsets_vec.push_back(0);  // First offset is always 0
        
        for (int64_t i = 1; i < num_bags; i++) {
            int64_t prev_offset = offsets_vec.back();
            int64_t max_increment = (num_indices - prev_offset) / (num_bags - i);
            int64_t increment = 0;
            if (tensor_offset < Size && max_increment > 0) {
                increment = static_cast<int64_t>(Data[tensor_offset++]) % (max_increment + 1);
            }
            offsets_vec.push_back(prev_offset + increment);
        }
        
        // If include_last_offset is true, add num_indices as the last offset
        if (include_last_offset) {
            offsets_vec.push_back(num_indices);
        }
        
        torch::Tensor offsets = torch::tensor(offsets_vec, torch::kInt64);
        
        // Set padding_idx if requested
        std::optional<int64_t> padding_idx = std::nullopt;
        if (padding_selector & 1) {
            padding_idx = static_cast<int64_t>(padding_selector % num_embeddings);
        }
        
        // Optional per_sample_weights
        std::optional<torch::Tensor> per_sample_weights = std::nullopt;
        if (use_per_sample_weights) {
            // per_sample_weights must have same shape as indices and be float
            per_sample_weights = torch::randn({num_indices}, torch::kFloat32);
        }
        
        // Inner try-catch for expected shape/type errors
        try {
            // Apply embedding_bag operation
            auto result = torch::embedding_bag(
                weight,
                indices,
                offsets,
                scale_grad_by_freq,
                mode,
                sparse,
                per_sample_weights,
                include_last_offset,
                padding_idx
            );
            
            // Access the result tuple to ensure computation happens
            auto output = std::get<0>(result);
            auto offset_out = std::get<1>(result);
            auto bag_size = std::get<2>(result);
            auto max_indices_out = std::get<3>(result);
            
            // Force computation
            (void)output.sum().item<float>();
        }
        catch (const c10::Error &e) {
            // Expected errors from invalid input combinations - catch silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}