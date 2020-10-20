/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2017, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     encmain.cpp
    \brief    Encoder application main
*/

#include <torch/torch.h>
#include <torch/script.h>
#include <time.h>
#include <iostream>
#include "TAppEncTop.h"
#include "TAppCommon/program_options_lite.h"

//! \ingroup TAppEncoder
//! \{

#include "../Lib/TLibCommon/Debug.h"

// Define a new Module.
//struct Net : torch::nn::Module {
//    Net() {
//        // Construct and register two Linear submodules.
//        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
//        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
//        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
//    }
//
//    // Implement the Net's algorithm.
//    torch::Tensor forward(torch::Tensor x) {
//        // Use one of many tensor manipulation functions.
//        x = torch::relu(fc1->forward(x.reshape({ x.size(0), 784 })));
//        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
//        x = torch::relu(fc2->forward(x));
//        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
//        return x;
//    }
//
//    // Use one of many "standard library" modules.
//    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
//};

// ====================================================================================================================
// Main function
// ====================================================================================================================

int main(int argc, char* argv[])
{
    //    // Create a new Net.
    //auto net = std::make_shared<Net>();

    //// Create a multi-threaded data loader for the MNIST dataset.
    //auto data_loader = torch::data::make_data_loader(
    //    torch::data::datasets::MNIST("../../mnist").map(
    //        torch::data::transforms::Stack<>()),
    //    /*batch_size=*/64);

    //// Instantiate an SGD optimization algorithm to update our Net's parameters.
    //torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    //for (size_t epoch = 1; epoch <= 10; ++epoch) {
    //    size_t batch_index = 0;
    //    // Iterate the data loader to yield batches from the dataset.
    //    for (auto& batch : *data_loader) {
    //        // Reset gradients.
    //        optimizer.zero_grad();
    //        // Execute the model on the input data.
    //        torch::Tensor prediction = net->forward(batch.data);
    //        // Compute a loss value to judge the prediction of our model.
    //        torch::Tensor loss = torch::nll_loss(prediction, batch.target);
    //        // Compute gradients of the loss w.r.t. the parameters of our model.
    //        loss.backward();
    //        // Update the parameters based on the calculated gradients.
    //        optimizer.step();
    //        // Output the loss and checkpoint every 100 batches.
    //        if (++batch_index % 100 == 0) {
    //            std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
    //                << " | Loss: " << loss.item<float>() << std::endl;
    //            // Serialize your model periodically as a checkpoint.
    //            torch::save(net, "net.pt");
    //        }
    //    }
    //}
  TAppEncTop  cTAppEncTop;

  /*torch::jit::script::Module pt_module;
  try {
      pt_module = torch::jit::load("traced_fractional_enhancement_model.pt");
  }
  catch (const c10::Error&) {
      std::cerr << "error loading the model\n";
      return -1;
  }

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({ 1, 3, 32, 32 }));
  
  at::Tensor output = pt_module.forward(inputs).toTensor();
  std::cout << output.sizes() << ":" << output.dim() << '\n';*/

  // print information
  fprintf( stdout, "\n" );
  fprintf( stdout, "HM software: Encoder Version [%s] (including RExt)", NV_VERSION );
  fprintf( stdout, NVM_ONOS );
  fprintf( stdout, NVM_COMPILEDBY );
  fprintf( stdout, NVM_BITS );
  fprintf( stdout, "\n\n" );

  // create application encoder class
  cTAppEncTop.create();

  // parse configuration
  try
  {
    if(!cTAppEncTop.parseCfg( argc, argv ))
    {
      cTAppEncTop.destroy();
#if ENVIRONMENT_VARIABLE_DEBUG_AND_TEST
      EnvVar::printEnvVar();
#endif
      return 1;
    }
  }
  catch (df::program_options_lite::ParseFailure &e)
  {
    std::cerr << "Error parsing option \""<< e.arg <<"\" with argument \""<< e.val <<"\"." << std::endl;
    return 1;
  }

#if PRINT_MACRO_VALUES
  printMacroSettings();
#endif

#if ENVIRONMENT_VARIABLE_DEBUG_AND_TEST
  EnvVar::printEnvVarInUse();
#endif

  // starting time
  Double dResult;
  clock_t lBefore = clock();

  // call encoding function
  cTAppEncTop.encode();

  // ending time
  dResult = (Double)(clock()-lBefore) / CLOCKS_PER_SEC;
  printf("\n Total Time: %12.3f sec.\n", dResult);

  // destroy application encoder class
  cTAppEncTop.destroy();

  return 0;
}

//! \}
