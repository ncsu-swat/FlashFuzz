#include "fuzzer_utils.h"
#include <iostream>
#include <algorithm>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Determine number of sequences (2-5)
        uint8_t num_sequences = (Data[offset++] % 4) + 2;
        
        // Determine feature size (shared across all sequences)
        int64_t feature_size = (Data[offset++] % 8) + 1;
        
        std::vector<torch::Tensor> sequences;
        std::vector<int64_t> lengths;
        
        // Create sequence tensors with same feature dimension but varying lengths
        for (uint8_t i = 0; i < num_sequences && offset + 4 < Size; ++i) {
            // Variable sequence length (1-16)
            int64_t seq_len = (Data[offset++] % 16) + 1;
            lengths.push_back(seq_len);
            
            // Create 2D tensor: [seq_len, feature_size]
            torch::Tensor seq = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape to proper dimensions
            int64_t total_elements = seq.numel();
            if (total_elements == 0) {
                total_elements = 1;
                seq = torch::zeros({1});
            }
            
            // Ensure we have a 2D tensor with [seq_len, feature_size]
            seq = seq.flatten().to(torch::kFloat32);
            int64_t needed = seq_len * feature_size;
            if (seq.numel() >= needed) {
                seq = seq.slice(0, 0, needed).reshape({seq_len, feature_size});
            } else {
                // Pad if needed
                auto padded = torch::zeros({needed}, torch::kFloat32);
                padded.slice(0, 0, seq.numel()).copy_(seq);
                seq = padded.reshape({seq_len, feature_size});
            }
            
            sequences.push_back(seq);
        }
        
        if (sequences.empty()) {
            return 0;
        }
        
        // Test pad_sequence
        try {
            bool batch_first = offset < Size && (Data[offset++] % 2 == 0);
            float padding_value = 0.0f;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&padding_value, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Sanitize padding value
                if (!std::isfinite(padding_value)) {
                    padding_value = 0.0f;
                }
            }
            
            torch::Tensor padded = torch::nn::utils::rnn::pad_sequence(sequences, batch_first, padding_value);
        } catch (const std::exception &) {
            // Expected for invalid configurations
        }
        
        // Test pack_sequence (requires sequences sorted by length in decreasing order)
        try {
            // Sort sequences by length (descending) for enforce_sorted=true
            std::vector<std::pair<int64_t, size_t>> len_idx;
            for (size_t i = 0; i < sequences.size(); ++i) {
                len_idx.push_back({sequences[i].size(0), i});
            }
            std::sort(len_idx.begin(), len_idx.end(), [](auto &a, auto &b) {
                return a.first > b.first;
            });
            
            std::vector<torch::Tensor> sorted_sequences;
            for (auto &p : len_idx) {
                sorted_sequences.push_back(sequences[p.second]);
            }
            
            bool enforce_sorted = offset < Size && (Data[offset++] % 2 == 0);
            
            auto packed = torch::nn::utils::rnn::pack_sequence(
                enforce_sorted ? sorted_sequences : sequences, 
                enforce_sorted
            );
            
            // Test pad_packed_sequence
            try {
                bool batch_first_unpack = offset < Size && (Data[offset++] % 2 == 0);
                float padding_value_unpack = 0.0f;
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&padding_value_unpack, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    if (!std::isfinite(padding_value_unpack)) {
                        padding_value_unpack = 0.0f;
                    }
                }
                
                c10::optional<int64_t> total_length = c10::nullopt;
                if (offset < Size && Data[offset++] % 2 == 0) {
                    // Find max length in sequences
                    int64_t max_len = 0;
                    for (const auto& seq : sequences) {
                        max_len = std::max(max_len, seq.size(0));
                    }
                    total_length = max_len + (Data[offset % Size] % 5);
                }
                
                auto unpacked = torch::nn::utils::rnn::pad_packed_sequence(
                    packed, 
                    batch_first_unpack, 
                    padding_value_unpack,
                    total_length
                );
            } catch (const std::exception &) {
                // Expected for invalid configurations
            }
        } catch (const std::exception &) {
            // Expected for invalid configurations
        }
        
        // Test pack_padded_sequence
        try {
            bool batch_first_pack = offset < Size && (Data[offset++] % 2 == 0);
            bool enforce_sorted_pack = offset < Size && (Data[offset++] % 2 == 0);
            
            // Create a padded sequence tensor
            // Find max length
            int64_t max_len = 0;
            for (const auto& seq : sequences) {
                max_len = std::max(max_len, seq.size(0));
            }
            
            int64_t batch_size = static_cast<int64_t>(sequences.size());
            
            // Create padded tensor
            torch::Tensor padded_input;
            if (batch_first_pack) {
                padded_input = torch::zeros({batch_size, max_len, feature_size}, torch::kFloat32);
                for (int64_t i = 0; i < batch_size; ++i) {
                    padded_input[i].slice(0, 0, sequences[i].size(0)).copy_(sequences[i]);
                }
            } else {
                padded_input = torch::zeros({max_len, batch_size, feature_size}, torch::kFloat32);
                for (int64_t i = 0; i < batch_size; ++i) {
                    padded_input.slice(1, i, i+1).slice(0, 0, sequences[i].size(0)).copy_(
                        sequences[i].unsqueeze(1)
                    );
                }
            }
            
            // Create lengths tensor - must be sorted descending if enforce_sorted
            torch::Tensor lengths_tensor = torch::tensor(lengths, torch::kLong);
            
            if (enforce_sorted_pack) {
                // Sort lengths in descending order and reorder padded_input accordingly
                auto sorted_result = lengths_tensor.sort(/*descending=*/true);
                lengths_tensor = std::get<0>(sorted_result);
                auto indices = std::get<1>(sorted_result);
                
                if (batch_first_pack) {
                    padded_input = padded_input.index_select(0, indices);
                } else {
                    padded_input = padded_input.index_select(1, indices);
                }
            }
            
            auto packed_padded = torch::nn::utils::rnn::pack_padded_sequence(
                padded_input, 
                lengths_tensor, 
                batch_first_pack, 
                enforce_sorted_pack
            );
        } catch (const std::exception &) {
            // Expected for invalid configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}