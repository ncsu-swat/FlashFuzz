#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

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
        
        // Create input tensors for RNN utilities
        torch::Tensor sequence;
        std::vector<torch::Tensor> sequences;
        std::vector<int64_t> lengths;
        
        // Create a sequence tensor
        sequence = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Determine number of sequences (1-5)
        uint8_t num_sequences = 1;
        if (offset < Size) {
            num_sequences = (Data[offset++] % 5) + 1;
        }
        
        // Create sequence tensors
        for (uint8_t i = 0; i < num_sequences && offset < Size; ++i) {
            torch::Tensor seq = fuzzer_utils::createTensor(Data, Size, offset);
            sequences.push_back(seq);
            
            // Generate a random length for this sequence
            int64_t length = 1;
            if (offset < Size) {
                length = static_cast<int64_t>(Data[offset++]) % 10 + 1;
                if (seq.dim() > 0 && seq.size(0) > 0) {
                    length = length % seq.size(0) + 1;
                }
            }
            lengths.push_back(length);
        }
        
        // Create a lengths tensor
        torch::Tensor lengths_tensor;
        if (!lengths.empty()) {
            lengths_tensor = torch::tensor(lengths, torch::kLong);
        } else {
            lengths_tensor = torch::ones({1}, torch::kLong);
        }
        
        // Test pad_sequence
        if (!sequences.empty()) {
            bool batch_first = offset < Size && (Data[offset++] % 2 == 0);
            float padding_value = 0.0f;
            if (offset < Size) {
                std::memcpy(&padding_value, Data + offset, std::min(sizeof(float), Size - offset));
                offset += std::min(sizeof(float), Size - offset);
            }
            
            torch::Tensor padded = torch::nn::utils::rnn::pad_sequence(sequences, batch_first, padding_value);
        }
        
        // Test pack_sequence
        if (!sequences.empty()) {
            bool enforce_sorted = offset < Size && (Data[offset++] % 2 == 0);
            auto packed = torch::nn::utils::rnn::pack_sequence(sequences, enforce_sorted);
            
            // Test pad_packed_sequence
            bool batch_first_unpack = offset < Size && (Data[offset++] % 2 == 0);
            float padding_value_unpack = 0.0f;
            if (offset < Size) {
                std::memcpy(&padding_value_unpack, Data + offset, std::min(sizeof(float), Size - offset));
                offset += std::min(sizeof(float), Size - offset);
            }
            
            torch::Tensor total_length_tensor;
            if (offset < Size) {
                int64_t total_length = static_cast<int64_t>(Data[offset++]) % 100 + 1;
                total_length_tensor = torch::tensor(total_length, torch::kLong);
            }
            
            auto unpacked = torch::nn::utils::rnn::pad_packed_sequence(
                packed, 
                batch_first_unpack, 
                padding_value_unpack,
                total_length_tensor.defined() ? total_length_tensor.item<int64_t>() : -1
            );
        }
        
        // Test pack_padded_sequence
        if (sequence.defined() && lengths_tensor.defined()) {
            bool batch_first_pack = offset < Size && (Data[offset++] % 2 == 0);
            bool enforce_sorted_pack = offset < Size && (Data[offset++] % 2 == 0);
            
            // Ensure sequence has at least 2 dimensions for pack_padded_sequence
            if (sequence.dim() < 2) {
                if (sequence.dim() == 0) {
                    sequence = sequence.unsqueeze(0).unsqueeze(0);
                } else {
                    sequence = sequence.unsqueeze(0);
                }
            }
            
            // Ensure lengths match batch dimension
            int64_t batch_dim = batch_first_pack ? 0 : 1;
            if (batch_dim < sequence.dim() && lengths_tensor.size(0) != sequence.size(batch_dim)) {
                if (lengths_tensor.size(0) > sequence.size(batch_dim)) {
                    lengths_tensor = lengths_tensor.slice(0, 0, sequence.size(batch_dim));
                } else {
                    // Pad lengths tensor
                    auto new_lengths = torch::ones({sequence.size(batch_dim)}, torch::kLong);
                    new_lengths.slice(0, 0, lengths_tensor.size(0)) = lengths_tensor;
                    lengths_tensor = new_lengths;
                }
            }
            
            // Ensure lengths are valid (not greater than sequence length)
            int64_t seq_dim = batch_first_pack ? 1 : 0;
            if (seq_dim < sequence.dim()) {
                int64_t max_length = sequence.size(seq_dim);
                for (int64_t i = 0; i < lengths_tensor.size(0); i++) {
                    if (lengths_tensor[i].item<int64_t>() > max_length) {
                        lengths_tensor[i] = max_length;
                    }
                }
            }
            
            auto packed_padded = torch::nn::utils::rnn::pack_padded_sequence(
                sequence, 
                lengths_tensor, 
                batch_first_pack, 
                enforce_sorted_pack
            );
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
