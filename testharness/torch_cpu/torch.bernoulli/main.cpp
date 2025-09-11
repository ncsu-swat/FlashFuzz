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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create probability tensor if we have more data
        torch::Tensor prob_tensor;
        if (offset + 4 < Size) {
            prob_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Clamp probability tensor values to [0, 1]
            if (prob_tensor.is_floating_point()) {
                prob_tensor = torch::clamp(prob_tensor, 0.0, 1.0);
            } else {
                // Convert to float and clamp if not floating point
                prob_tensor = torch::clamp(prob_tensor.to(torch::kFloat), 0.0, 1.0);
            }
        }
        
        // Get a single probability value if we have more data
        double prob_value = 0.5; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&prob_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Clamp probability value to [0, 1]
            prob_value = std::max(0.0, std::min(1.0, prob_value));
        }
        
        // Try different variants of bernoulli
        torch::Tensor result1, result2, result3;
        
        // Variant 1: bernoulli with probability tensor
        if (prob_tensor.defined()) {
            result1 = torch::bernoulli(input_tensor, prob_tensor);
        }
        
        // Variant 2: bernoulli with probability value
        result2 = torch::bernoulli(input_tensor, prob_value);
        
        // Variant 3: in-place bernoulli
        if (input_tensor.is_floating_point() || input_tensor.scalar_type() == torch::kBool) {
            result3 = input_tensor.clone();
            result3.bernoulli_(prob_value);
        }
        
        // Variant 4: in-place bernoulli with generator
        if (offset < Size && (input_tensor.is_floating_point() || input_tensor.scalar_type() == torch::kBool)) {
            torch::Tensor result4 = input_tensor.clone();
            auto gen = torch::default_generator();
            uint64_t seed = 0;
            if (offset + sizeof(uint64_t) <= Size) {
                std::memcpy(&seed, Data + offset, sizeof(uint64_t));
                offset += sizeof(uint64_t);
            }
            gen.set_current_seed(seed);
            result4.bernoulli_(prob_value, gen);
        }
        
        // Variant 5: bernoulli with generator
        if (offset < Size) {
            auto gen = torch::default_generator();
            uint64_t seed = 0;
            if (offset + sizeof(uint64_t) <= Size) {
                std::memcpy(&seed, Data + offset, sizeof(uint64_t));
                offset += sizeof(uint64_t);
            }
            gen.set_current_seed(seed);
            torch::bernoulli(input_tensor, prob_value, gen);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
