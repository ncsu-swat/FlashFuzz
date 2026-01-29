#include "fuzzer_utils.h"
#include <iostream>
#include <torch/script.h>

// --- Fuzzer Entry Point ---
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

        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Create a second tensor if we have more data
        torch::Tensor second_tensor;
        if (offset + 4 < Size) {
            second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            second_tensor = input_tensor.clone();
        }

        // Get a variant selector if we have data left
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 8;
        }

        // Test different JIT compilation and execution scenarios
        try {
            switch (variant) {
                case 0: {
                    // Simple doubling function
                    auto cu = torch::jit::compile(R"(
                        def forward(x):
                            return x * 2
                    )");
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    auto output = cu->get_function("forward")(inputs);
                    if (output.isTensor()) {
                        torch::Tensor result = output.toTensor();
                    }
                    break;
                }

                case 1: {
                    // Two-input function
                    auto cu = torch::jit::compile(R"(
                        def forward(x, y):
                            return x + y
                    )");
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    inputs.push_back(second_tensor);
                    auto output = cu->get_function("forward")(inputs);
                    if (output.isTensor()) {
                        torch::Tensor result = output.toTensor();
                    }
                    break;
                }

                case 2: {
                    // Control flow with dimension check
                    auto cu = torch::jit::compile(R"(
                        def forward(x):
                            if x.dim() > 0:
                                return x.sum()
                            return x
                    )");
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    auto output = cu->get_function("forward")(inputs);
                    break;
                }

                case 3: {
                    // Numerical operations with epsilon
                    auto cu = torch::jit::compile(R"(
                        def forward(x):
                            return 1.0 / (x.abs() + 0.0001)
                    )");
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    auto output = cu->get_function("forward")(inputs);
                    break;
                }

                case 4: {
                    // Loop in TorchScript
                    auto cu = torch::jit::compile(R"(
                        def forward(x):
                            result = x
                            for i in range(3):
                                result = result + x
                            return result
                    )");
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    auto output = cu->get_function("forward")(inputs);
                    break;
                }

                case 5: {
                    // Multiple tensor operations
                    auto cu = torch::jit::compile(R"(
                        def forward(x):
                            y = x.relu()
                            z = y.sigmoid()
                            return z * x
                    )");
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    auto output = cu->get_function("forward")(inputs);
                    break;
                }

                case 6: {
                    // Conditional with numel check
                    auto cu = torch::jit::compile(R"(
                        def forward(x):
                            if x.numel() > 0:
                                return x.mean()
                            return x
                    )");
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    auto output = cu->get_function("forward")(inputs);
                    break;
                }

                case 7: {
                    // Flatten and reshape operations
                    auto cu = torch::jit::compile(R"(
                        def forward(x):
                            if x.dim() > 1:
                                return x.flatten()
                            return x
                    )");
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(input_tensor);
                    auto output = cu->get_function("forward")(inputs);
                    break;
                }
            }
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors (shape mismatches, etc.)
        } catch (const std::runtime_error& e) {
            // Catch JIT compilation/execution errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}