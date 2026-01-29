#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Create input tensor - this will be used as probability values
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float and clamp to [0, 1] for valid probabilities
        torch::Tensor prob_tensor;
        if (input_tensor.is_floating_point()) {
            prob_tensor = torch::clamp(input_tensor, 0.0, 1.0);
        } else {
            prob_tensor = torch::clamp(input_tensor.to(torch::kFloat), 0.0, 1.0);
        }
        
        // Get a single probability value for scalar variants
        double prob_value = 0.5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&prob_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Clamp probability value to [0, 1]
            if (std::isnan(prob_value) || std::isinf(prob_value)) {
                prob_value = 0.5;
            } else {
                prob_value = std::max(0.0, std::min(1.0, prob_value));
            }
        }
        
        // Variant 1: bernoulli(input) - input is the probability tensor
        // torch::bernoulli samples from Bernoulli distribution where input contains probabilities
        torch::Tensor result1 = torch::bernoulli(prob_tensor);
        
        // Variant 2: bernoulli with generator
        if (offset + sizeof(uint64_t) <= Size) {
            auto gen = torch::make_generator<torch::CPUGeneratorImpl>();
            uint64_t seed = 0;
            std::memcpy(&seed, Data + offset, sizeof(uint64_t));
            offset += sizeof(uint64_t);
            gen.set_current_seed(seed);
            torch::Tensor result2 = torch::bernoulli(prob_tensor, gen);
        }
        
        // Variant 3: in-place bernoulli_(p) with scalar probability
        // bernoulli_ fills tensor with values from Bernoulli(p)
        {
            torch::Tensor result3 = torch::empty_like(prob_tensor);
            result3.bernoulli_(prob_value);
        }
        
        // Variant 4: in-place bernoulli_ with generator
        if (offset + sizeof(uint64_t) <= Size) {
            auto gen = torch::make_generator<torch::CPUGeneratorImpl>();
            uint64_t seed = 0;
            std::memcpy(&seed, Data + offset, sizeof(uint64_t));
            offset += sizeof(uint64_t);
            gen.set_current_seed(seed);
            
            torch::Tensor result4 = torch::empty_like(prob_tensor);
            result4.bernoulli_(prob_value, gen);
        }
        
        // Variant 5: in-place bernoulli_ using another tensor as probabilities
        // bernoulli_(src) fills self with values sampled from Bernoulli(src)
        {
            torch::Tensor result5 = torch::empty_like(prob_tensor);
            result5.bernoulli_(prob_tensor);
        }
        
        // Variant 6: Test with different tensor shapes
        if (offset + 2 <= Size) {
            int dim1 = (Data[offset] % 10) + 1;
            int dim2 = (Data[offset + 1] % 10) + 1;
            offset += 2;
            
            torch::Tensor shaped_probs = torch::rand({dim1, dim2});
            torch::Tensor result6 = torch::bernoulli(shaped_probs);
        }
        
        // Variant 7: Test with specific probability values (edge cases)
        {
            // All zeros probability
            torch::Tensor zeros = torch::zeros({3, 3});
            torch::Tensor result7a = torch::bernoulli(zeros);
            
            // All ones probability  
            torch::Tensor ones = torch::ones({3, 3});
            torch::Tensor result7b = torch::bernoulli(ones);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}