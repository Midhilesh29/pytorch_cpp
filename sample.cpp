#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <string>
#include <iostream>
#include <memory>
#include <vector>

const char* kDataRoot = "../data";
const int64_t kTestBatchSize = 1000;
const int64_t kNumberOfEpochs = 10;

template <typename DataLoader>

void test(torch::jit::script::Module model,torch::Device device, DataLoader& data_loader, size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    std::vector<torch::jit::IValue> input;
    input.push_back(data);
    auto output = model.forward(input).toTensor();
    std::cout<<output<<endl;
    //auto pred = output.argmax(1);
    //correct += pred.eq(targets).sum().template item<int64_t>();
  }
  /*
  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Accuracy: %.3f\n",
      static_cast<double>(correct) / dataset_size);
  */
}

int main(int argc, const char* argv[]) {

  if(argc!=2){
  	std::cerr << "usage: example-app <path-to-exported-script-module>\n";
  	return -1;
  }

  torch::manual_seed(1);
  torch::DeviceType device_type = torch::kCUDA;
  torch::Device device(device_type);
  torch::jit::script::Module module;
  module = torch::jit::load(argv[1]);
//module.to(device);

  auto test_dataset = torch::data::datasets::MNIST(
            kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();

  std::cout<<"Data size:"<<test_dataset_size<<std::endl;
  auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);
  test(module, device, *test_loader, test_dataset_size);

/*

  try{
    module = torch::jit::load(argv[1]);
    //module.to(device);

    auto test_dataset = torch::data::datasets::MNIST(
              kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
              .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
              .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();

    std::cout<<"Data size:"<<test_dataset_size;
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);
    test(module, device, *test_loader, test_dataset_size);
  }
  catch (const c10::Error& e){
  	std::cerr << "error loading the model\n";
  	return -1;
  }*/
}