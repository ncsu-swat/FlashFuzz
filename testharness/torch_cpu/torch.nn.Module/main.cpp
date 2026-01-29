#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>
#include <torch/torch.h>

class SimpleModule : public torch::nn::Module {
public:
    SimpleModule() {
        linear1 = register_module("linear1", torch::nn::Linear(10, 8));
        linear2 = register_module("linear2", torch::nn::Linear(8, 4));
        dropout = register_module("dropout", torch::nn::Dropout(0.5));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(linear1->forward(x));
        x = dropout->forward(x);
        x = linear2->forward(x);
        return torch::log_softmax(x, 1);
    }

    torch::nn::Linear linear1{nullptr}, linear2{nullptr};
    torch::nn::Dropout dropout{nullptr};
};

class ConvModule : public torch::nn::Module {
public:
    ConvModule() {
        conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1).padding(1)));
        bn = register_module("bn", torch::nn::BatchNorm2d(16));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn->forward(conv->forward(x)));
        return torch::max_pool2d(x, 2);
    }

    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
};

class RecurrentModule : public torch::nn::Module {
public:
    RecurrentModule() {
        lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(10, 20).num_layers(2).bidirectional(true)));
        linear = register_module("linear", torch::nn::Linear(40, 5));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto lstm_out = lstm->forward(x);
        auto output = std::get<0>(lstm_out);
        auto last_output = output.select(0, output.size(0) - 1);
        return linear->forward(last_output);
    }

    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear linear{nullptr};
};

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
        
        if (Size < 2) return 0;
        
        uint8_t module_type = Data[offset++] % 3;
        uint8_t batch_size = (Data[offset++] % 10) + 1;
        
        torch::Tensor input;
        torch::Tensor output;
        
        switch (module_type) {
            case 0: {
                auto simple_module = std::make_shared<SimpleModule>();
                
                // Create input tensor with shape [batch_size, 10]
                try {
                    if (offset < Size) {
                        input = fuzzer_utils::createTensor(Data, Size, offset);
                        // Flatten and reshape to required dimensions
                        input = input.flatten();
                        int64_t total = input.numel();
                        if (total >= 10) {
                            int64_t batches = total / 10;
                            if (batches > 0) {
                                input = input.narrow(0, 0, batches * 10).reshape({batches, 10});
                            } else {
                                input = torch::randn({batch_size, 10});
                            }
                        } else {
                            input = torch::randn({batch_size, 10});
                        }
                    } else {
                        input = torch::randn({batch_size, 10});
                    }
                } catch (...) {
                    input = torch::randn({batch_size, 10});
                }
                
                {
                    torch::NoGradGuard no_grad;
                    output = simple_module->forward(input);
                    
                    if (offset < Size) {
                        bool train_mode = Data[offset++] % 2 == 1;
                        if (train_mode) {
                            simple_module->train();
                        } else {
                            simple_module->eval();
                        }
                        
                        auto output2 = simple_module->forward(input);
                        
                        if (train_mode) {
                            simple_module->dropout->eval();
                            auto output3 = simple_module->forward(input);
                        }
                    }
                    
                    // Test parameter access
                    auto params = simple_module->parameters();
                    auto named_params = simple_module->named_parameters();
                    
                    // Test module cloning
                    auto cloned = simple_module->clone();
                    
                    // Test state_dict related methods
                    simple_module->zero_grad();
                }
                break;
            }
            case 1: {
                auto conv_module = std::make_shared<ConvModule>();
                
                // Create input tensor with shape [batch_size, 3, H, W]
                int64_t height = 32, width = 32;
                if (offset + 1 < Size) {
                    height = (Data[offset++] % 32) + 8;  // 8-39
                    width = (Data[offset++] % 32) + 8;   // 8-39
                }
                
                try {
                    if (offset < Size) {
                        input = fuzzer_utils::createTensor(Data, Size, offset);
                        input = input.flatten();
                        int64_t total = input.numel();
                        int64_t required = 3 * height * width;
                        if (total >= required) {
                            int64_t batches = total / required;
                            if (batches > 0) {
                                input = input.narrow(0, 0, batches * required).reshape({batches, 3, height, width});
                            } else {
                                input = torch::randn({batch_size, 3, height, width});
                            }
                        } else {
                            input = torch::randn({batch_size, 3, height, width});
                        }
                    } else {
                        input = torch::randn({batch_size, 3, height, width});
                    }
                } catch (...) {
                    input = torch::randn({batch_size, 3, height, width});
                }
                
                {
                    torch::NoGradGuard no_grad;
                    output = conv_module->forward(input);
                    
                    if (offset < Size) {
                        bool train_mode = Data[offset++] % 2 == 1;
                        if (train_mode) {
                            conv_module->train();
                        } else {
                            conv_module->eval();
                        }
                        
                        auto output2 = conv_module->forward(input);
                    }
                    
                    // Test buffers (BatchNorm has running_mean, running_var)
                    auto buffers = conv_module->buffers();
                    auto named_buffers = conv_module->named_buffers();
                    
                    // Test children and modules
                    auto children = conv_module->children();
                    auto modules = conv_module->modules();
                    auto named_modules = conv_module->named_modules();
                }
                break;
            }
            case 2: {
                auto rnn_module = std::make_shared<RecurrentModule>();
                
                // Create input tensor with shape [seq_len, batch_size, 10]
                int64_t seq_len = 5;
                if (offset < Size) {
                    seq_len = (Data[offset++] % 10) + 1;  // 1-10
                }
                
                try {
                    if (offset < Size) {
                        input = fuzzer_utils::createTensor(Data, Size, offset);
                        input = input.flatten();
                        int64_t total = input.numel();
                        int64_t required = seq_len * 10;
                        if (total >= required) {
                            int64_t batches = total / required;
                            if (batches > 0) {
                                input = input.narrow(0, 0, batches * required).reshape({seq_len, batches, 10});
                            } else {
                                input = torch::randn({seq_len, batch_size, 10});
                            }
                        } else {
                            input = torch::randn({seq_len, batch_size, 10});
                        }
                    } else {
                        input = torch::randn({seq_len, batch_size, 10});
                    }
                } catch (...) {
                    input = torch::randn({seq_len, batch_size, 10});
                }
                
                {
                    torch::NoGradGuard no_grad;
                    output = rnn_module->forward(input);
                    
                    if (offset < Size) {
                        bool train_mode = Data[offset++] % 2 == 1;
                        if (train_mode) {
                            rnn_module->train();
                        } else {
                            rnn_module->eval();
                        }
                        
                        auto output2 = rnn_module->forward(input);
                    }
                    
                    // Test apply function
                    rnn_module->apply([](torch::nn::Module& m) {
                        // Just iterate through modules
                    });
                    
                    // Test is_training
                    bool is_training = rnn_module->is_training();
                }
                break;
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