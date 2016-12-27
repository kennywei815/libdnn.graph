// Copyright 2013-2014 [Author: Po-Wei Chou]
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <set>
#include <dnn.h>
#include <dnn-graph.h>
#include <dnn-utility.h>
#include <cmdparser.h>
#include <rbm.h>
#include <batch.h>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cstring>
using namespace std;

struct TrainConfig {
  string train_program;

  size_t graph_id;  //test

  string train_fn;

  //string model_in; //test
  string input_net_in; //test
  string hidden_net_in; //test
  string out_net_in; //test
  vector<string> path;

  string model_out;

  size_t input_dim;
  NormType n_type;
  string n_filename;
  int base;

  int ratio;
  size_t batchSize;
  float learningRate;
  float objectiveWeight;
  float minValidAcc;
  size_t maxEpoch;
  string layer;
  //vector<bool> layerOn;

  size_t cache_size;
};

vector<size_t> dnn_predict(DNNGraph& dnnGraph, vector<DataSet*>& data_vec, ERROR_MEASURE errorMeasure);
void dnn_train(DNNGraph& dnnGraph, vector<DataSet*>& train_vec, vector<DataSet*>& valid_vec, ERROR_MEASURE errorMeasure);
void dnn_train_switch_in_minibatch_level(vector<DNNGraph>& dnnGraph_vec, vector<vector<DataSet*>>& train_vec, vector<vector<DataSet*>>& valid_vec, ERROR_MEASURE errorMeasure);
bool isEoutStopDecrease(const std::vector<size_t> Eout, size_t epoch, size_t nNonIncEpoch);
void parse_arg(string parms, Config& config, TrainConfig& setting);

int main (int argc, char* argv[]) {
  if (argc != 3 || strcmp(argv[1], "-f") != 0) {
    cerr << "Usage: dnn-train.mod -f <script_file>\n";
    return 1;
  }

  
  ifstream              script_file(argv[2]);
  string                parms;
  int                   num_network = 0;
  vector<Config>        config_vec;
  vector<TrainConfig>   setting_vec; //training parms
  Config                config;
  TrainConfig           setting; //training parms
  int                   num_iter;



  getline(script_file, parms);
  //read in parameters not belonging to dnn-train #TODO: correct this sentence to formal English!
  //TODO: refactor here
  if (script_file) {
    stringstream ss(parms);
    string parm_name;

    cout << parms << endl; //test
    ss >> parm_name;
    if (parm_name == "iteration") {
      ss >> num_iter;
      cout << "iteration " << num_iter << endl;
    }
    else {
      cerr << "ERROR: Firt non-comment line should be \"iteration <iteration>\"\n";
      // usage() //TODO
      exit(1);
    }
    getline(script_file, parms);
  }
  while (script_file) {
    num_network++;
    parse_arg(parms, config, setting);
    config_vec.push_back(config);
    setting_vec.push_back(setting);

    getline(script_file, parms);
  }
  if (num_network == 0) {
    cerr << "ERROR: script file is empty!\n";
    return 1;
  }

  cout << setting.cache_size << endl; //test
  CudaMemManager<float>::setCacheSize( setting.cache_size ); //test
  
  // Set up graphs
  vector<size_t> pathID2graphID( num_network );
  size_t num_graph = setting_vec.back().graph_id + 1; //test
  for(size_t i=0; i<num_network; i++)
    pathID2graphID[i] = setting_vec[i].graph_id;


  // Load model
  vector<DNN> dnn_vec( num_network );
  vector< vector<DNN*> > graph_dnn_vec( num_graph ); //test
  vector< vector<size_t> > layer_spec_vec( num_network );

  map<string, DNN> sub_net_pool; //test

  for (int i=0; i<num_network; i++) {
    // [DONE] need to support any number of subnets
    for (size_t j=0; j<setting_vec[i].path.size(); j++) {
      if ( sub_net_pool.find(setting_vec[i].path[j]) == sub_net_pool.end() ) { //new sub_net
        sub_net_pool.insert( pair<string, DNN>(setting_vec[i].path[j], DNN(setting_vec[i].path[j])) );
      }
    }
  }
  //TODO: refactoring
  //fix dnn_vec memory address to prevent using too much cache space. (CudaMemManager IMPLEMENTS A CACHE WITH CACHE COHERENCE IN MIND.)

  vector<DNNGraph> dnnGraph_vec( num_graph );
  vector<size_t>   batchSize_vec( num_graph );
  //DNNGraph dnnGraph;

  for (int i=0; i<num_network; i++) {
    // [DONE] need to support any number of subnets
    // TODO: remove dnn_vec???
    for (size_t j=0; j<setting_vec[i].path.size(); j++) {
      dnn_vec[i].append( sub_net_pool[ setting_vec[i].path[j] ] );
    }

    auto& t = dnn_vec[i].getTransforms();
    if (setting_vec[i].layer == "all") {
      config_vec[i].layerOn = vector<bool>(t.size(), true);
    }
    else {
      vector<size_t> layerIds = splitAsInt(setting_vec[i].layer, ':');
      config_vec[i].layerOn = vector<bool>(t.size(), false);
      for (auto l : layerIds)
        config_vec[i].layerOn[l] = true;
    }

    dnn_vec[i].setConfig( config_vec[i] );
    // end TODO


    vector<DNN*>   path;
    vector<string> names;
    for (size_t j=0; j<setting_vec[i].path.size(); j++) {
      path.push_back( &sub_net_pool[ setting_vec[i].path[j] ] );
      names.push_back( setting_vec[i].path[j] );
    }

    size_t graph_id = pathID2graphID[i];
    cout << "graph id: " << graph_id << endl; //test
    dnnGraph_vec[graph_id].addPath(path, names, config_vec[i].layerOn, setting_vec[i].objectiveWeight);
    dnnGraph_vec[graph_id].setConfig(config_vec[i]); //test

    graph_dnn_vec[graph_id].push_back(&dnn_vec[i]);
  }
  cout << "num_graph: " << num_graph << endl; //test
  //cout << "size(g_dnn_vec[0]): " << graph_dnn_vec[0].size() << endl; //test
  //cout << "size(g_dnn_vec[1]): " << graph_dnn_vec[1].size() << endl; //test
  //graph_dnn_vec[0][0]->getConfig().print();
  //graph_dnn_vec[1][0]->getConfig().print();

  // Load data
  vector<DataSet> data_vec( num_network );
  vector<DataSet> train_vec( num_network ), valid_vec( num_network );

  vector< vector<DataSet*> > graph_data_vec( num_graph );
  vector< vector<DataSet*> > graph_train_vec( num_graph ), graph_valid_vec( num_graph );

  // TODO: implement full data-dnnGraph mapping
  for (int i=0; i<num_network; i++) {
    size_t graph_id = pathID2graphID[i];

    data_vec[i] = DataSet(setting_vec[i].train_fn, setting_vec[i].input_dim, setting_vec[i].base);
    // data_vec[i].loadPrecomputedStatistics(n_filename);
    data_vec[i].setNormType(setting_vec[i].n_type);
    data_vec[i].showSummary();
    DataSet::split(data_vec[i], train_vec[i], valid_vec[i], setting_vec[i].ratio);
    config_vec[i].print();

    graph_data_vec[graph_id] .push_back(&data_vec[i]);
    graph_train_vec[graph_id].push_back(&train_vec[i]);
    graph_valid_vec[graph_id].push_back(&valid_vec[i]);
  }

  // Start Training
  ERROR_MEASURE err = CROSS_ENTROPY;
  for (int iter=0; iter<num_iter; iter++) {
    cout << "\n" << "iteration [" << iter << "]" << endl;
    /*
    for (int graph_id=0; graph_id<num_graph; graph_id++) {
      cout << "graph_id [" << graph_id << "]" << endl;
      //dnn_train(dnnGraph_vec[graph_id], graph_train_vec[graph_id], graph_valid_vec[graph_id], err);
      dnn_train(dnnGraph_vec[graph_id], graph_train_vec[graph_id], graph_valid_vec[graph_id], err); //test
    }
    */
    dnn_train_switch_in_minibatch_level(dnnGraph_vec, graph_train_vec, graph_valid_vec, err);
  }

  // Save the model
  for (int i=0; i<num_network; i++) {
    if (setting_vec[i].model_out.empty())
      setting_vec[i].model_out = setting_vec[i].train_fn.substr(setting_vec[i].train_fn.find_last_of('/') + 1) + ".model";

    dnn_vec[i].save(setting_vec[i].model_out);
  }

  return 0;
}


void parse_arg(string parms, Config& config, TrainConfig& setting) {
  stringstream ss(parms);
  int tmp_argc = 0;
  char** tmp_argv;
  string token;
  vector<string> tokens;

  ss >> token;
  while(ss) {
    tmp_argc++;
    tokens.push_back(token);
    ss >> token;
  }

  for(int i=0; i<tokens.size(); i++) {
    cout << tokens[i] << " ";
  }
  cout << endl;
  cout << tmp_argc << endl;

  tmp_argv = new char*[tmp_argc]  ;
  for(int i=0; i<tokens.size(); i++) {
    tmp_argv[i] = (char*)tokens[i].c_str();
  }


  CmdParser cmd(tmp_argc, tmp_argv);

  //TODO: need to support any number of subnets
  cmd.add("graph_id")
     .add("training_set_file") //.add("model_in")
     .add("input_net_in")
     .add("hidden_net_in")
     .add("out_net_in")
     .add("model_out", false);

  cmd.addGroup("Feature options:")
     .add("--input-dim", "specify the input dimension (dimension of feature).\n"
         "0 for auto detection.")
     .add("--normalize", "Feature normalization: \n"
        "0 -- Do not normalize.\n"
        "1 -- Rescale each dimension to [0, 1] respectively.\n"
        "2 -- Normalize to standard score. z = (x-u)/sigma .", "0")
     .add("--nf", "Load pre-computed statistics from file", "")
     .add("--base", "Label id starts from 0 or 1 ?", "0");

  cmd.addGroup("Training options: ")
     .add("-v", "ratio of training set to validation set (split automatically)", "5")
     .add("--max-epoch", "number of maximum epochs", "100000")
     .add("--min-acc", "Specify the minimum cross-validation accuracy", "0.5")
     .add("--learning-rate", "learning rate in back-propagation", "0.1")
     .add("--batch-size", "number of data per mini-batch", "32")
     .add("--objective-weight", "weight for this objective", "1")
     .add("--layer", "Specify which layers to copy (or dump). "
         "Ex: 1:2:9 means only print out layer 1, 2 and 9.", "all"); //New

  cmd.addGroup("Hardward options:")
     .add("--cache", "specify cache size (in MB) in GPU used by cuda matrix.", "16");

  cmd.addGroup("Example usage: dnn-train data/train3.dat --nodes=16-8");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  //TODO: modify ../tools/utility/include/cmdparser.h to have cmd[0]
  string train_program = tmp_argv[0];

  //TODO: modify ../tools/utility/include/cmdparser.h to have integer arguments
  size_t graph_id      = int(tmp_argv[1][0]) - int('a');
  cout << "graph id: " << graph_id << endl; //test

  string train_fn      = cmd[2];

  //TODO: need to support any number of subnets
  vector<string>  path;

  string input_net_in  = cmd[3];
  string hidden_net_in = cmd[4];
  string out_net_in    = cmd[5];

  path.push_back(input_net_in);
  path.push_back(hidden_net_in);
  path.push_back(out_net_in);

  string model_out      = cmd[6];

  size_t input_dim      = cmd["--input-dim"];
  NormType n_type       = (NormType) (int) cmd["--normalize"];
  string n_filename     = cmd["--nf"];
  int base              = cmd["--base"];

  int ratio             = cmd["-v"];
  size_t batchSize      = cmd["--batch-size"];
  float learningRate    = cmd["--learning-rate"];
  float objectiveWeight = cmd["--objective-weight"];
  float minValidAcc     = cmd["--min-acc"];
  size_t maxEpoch       = cmd["--max-epoch"];
  string layer          = cmd["--layer"];

  size_t cache_size     = cmd["--cache"];

  setting.train_program   = train_program;
  setting.graph_id        = graph_id;
  setting.train_fn        = train_fn;
  setting.path            = path;     
  setting.input_net_in    = input_net_in;     
  setting.hidden_net_in   = hidden_net_in;     
  setting.out_net_in      = out_net_in;     
  setting.model_out       = model_out;

  setting.input_dim       = input_dim;
  setting.n_type          = n_type;
  setting.n_filename      = n_filename;
  setting.base            = base;

  setting.ratio           = ratio;
  setting.batchSize       = batchSize;
  setting.learningRate    = learningRate;
  setting.objectiveWeight = objectiveWeight;
  setting.minValidAcc     = minValidAcc;
  setting.maxEpoch        = maxEpoch;
  setting.layer           = layer;

  setting.cache_size      = cache_size;

  //CudaMemManager<float>::setCacheSize(cache_size);

  // Set configurations
  config.learningRate = learningRate;
  config.maxEpoch = maxEpoch;
  config.batchSize = batchSize;
  config.trainValidRatio = ratio;
  config.minValidAccuracy = minValidAcc;
  /*
  */

  delete tmp_argv;
}


//TODO: train_label_vec, train_feat_vec...
void dnn_train(DNNGraph& dnnGraph, vector<DataSet*>& train_vec, vector<DataSet*>& valid_vec, ERROR_MEASURE errorMeasure) {
  size_t MAX_EPOCH = dnnGraph.getConfig().maxEpoch;
  size_t epoch;
  vector<vector<size_t>> Ein;
  vector<vector<size_t>> Eout;
  size_t batchSize = dnnGraph.getConfig().batchSize;

  float lr = dnnGraph.getConfig().learningRate / batchSize;

  size_t nTrain = train_vec.front()->size(),
         nValid = valid_vec.front()->size();

  mat fout;

  printf("Training...");
  fflush(stdout);
  perf::Timer timer;
  timer.start();
  for (epoch=0; epoch<MAX_EPOCH; ++epoch) {
    printf(" %lu", epoch);
    fflush(stdout);

    Batches batches(batchSize, nTrain);
    for (auto itr = batches.begin(); itr != batches.end(); ++itr) {
      /*
      //test
      DataSet* train = train_vec[0];
      DNN* dnn = dnn_vec[0];
      auto layerOn = dnn->getConfig().layerOn;

      auto data = (*train)[itr];

      dnn->feedForward(fout, data.x);

      mat error = getError( data.y, fout, errorMeasure);

      dnn->backPropagate(error, data.x, fout, lr, layerOn);
      */

      // Copy a batch of data from host to device
      vector<BatchData> data_vec;
      data_vec.clear();
      for (auto train : train_vec) {
        auto batch = (*train)[itr];
        data_vec.push_back( batch );
      }

      dnnGraph.feedForward(data_vec[0].x);
      dnnGraph.backPropagate(data_vec, lr, errorMeasure);
    }

    //Ein.push_back( dnn_predict(dnnGraph, train_vec, errorMeasure) );
    //Eout.push_back( dnn_predict(dnnGraph, valid_vec, errorMeasure) );
  }
  printf("\n");
  timer.elapsed();
  timer.reset();
  fflush(stdout);

  /*
  // output messages
  printf("Predicting...\n");
  timer.start();
  for (int i=0; i<Ein[0].size(); i++) {
    printf("._______._________________________._________________________.\n"
           "|       |                         |                         |\n"
           "|       |        In-Sample        |      Out-of-Sample      |\n"
           "| Epoch |__________.______________|__________.______________|\n"
           "|       |          |              |          |              |\n"
           "|       | Accuracy | # of correct | Accuracy | # of correct |\n"
           "|_______|__________|______________|__________|______________|\n");
    fflush(stdout);
    for (epoch=0; epoch<MAX_EPOCH; ++epoch) {
  
      float train_Acc = 1.0f - (float) Ein[epoch][i] / nTrain;
  
      if (train_Acc < 0) {
          cout << "."; cout.flush();
          continue;
      }
  
      float valid_Acc = 1.0f - (float) Eout[epoch][i] / nValid;
  
      printf("|%4lu   |  %5.2f %% |  %7lu     |  %5.2f %% |  %7lu     |\n",
        epoch, train_Acc * 100, nTrain - Ein[epoch][i], valid_Acc * 100, nValid - Eout[epoch][i]);
    }  
  }
  */
  // Show Summary
  printf("\n%ld epochs in total\n", epoch);
  timer.elapsed();
  timer.reset();
  fflush(stdout);
  //dnn.adjustLearningRate(trainAcc);
}

void dnn_train_switch_in_minibatch_level(vector<DNNGraph>& dnnGraph_vec, vector<vector<DataSet*>>& train_vec, vector<vector<DataSet*>>& valid_vec, ERROR_MEASURE errorMeasure) {
  size_t num_network = dnnGraph_vec.size();

  size_t epoch; //test

  vector<size_t> batchSize_vec(num_network);
  vector<size_t> MAX_EPOCH_vec(num_network);
  vector<float>  lr_vec       (num_network);
  vector<size_t> nTrain_vec   (num_network);
  vector<size_t> nValid_vec   (num_network);

  printf("Training...\n");
  fflush(stdout);
  perf::Timer timer;
  timer.start();

  for (int i=0; i<num_network; i++) {
    batchSize_vec[i] = dnnGraph_vec[i].getConfig().batchSize;
    MAX_EPOCH_vec[i] = dnnGraph_vec[i].getConfig().maxEpoch;
    lr_vec[i] = dnnGraph_vec[i].getConfig().learningRate / batchSize_vec[i];
  
    nTrain_vec[i] = train_vec[i][0]->size(),
    nValid_vec[i] = valid_vec[i][0]->size();

    //cerr << "nTrain_vec[i] = " << nTrain_vec[i] << endl; //test
  }


  
  //Batches
  vector<Batches> batches_vec;
  batches_vec.reserve(num_network);

  for (int i=0; i<num_network; i++) {
    batches_vec.push_back( Batches(batchSize_vec[i], nTrain_vec[i]) );
  }

  vector<Batches::iterator> itr_vec;
  itr_vec.reserve(num_network);
  for (int i=0; i<num_network; i++) {
    itr_vec.push_back(Batches::iterator( batches_vec[i].begin() ));
  }

  //shuffle mini-batches
  // [DONE] TODO: implement MAX_EPOCH_vec function by add seq[i] MAX_EPOCH_vec[i] times  and  check the iterator for end()
  vector<size_t> train_seq;
  for (int i=0; i<num_network; i++) {
    //vector<size_t> seq(batches_vec[i].size(), i); //fall back
    vector<size_t> seq(batches_vec[i].size() * MAX_EPOCH_vec[i], i);
    train_seq.insert( train_seq.end(), seq.begin(), seq.end() );
  }

  random_shuffle (train_seq.begin(), train_seq.end());
  
  //training
  Batches::iterator itr = batches_vec[0].end(); //iterators have to be initialized first!
  //TODO: progress bar
  for (int idx=0; idx<train_seq.size(); ++idx) {
    //cerr << idx << " "; //test
    size_t net_id = train_seq[idx];
    if (itr_vec[net_id] == batches_vec[net_id].end())
      itr_vec[net_id] = batches_vec[net_id].begin();

    itr = itr_vec[net_id];

    // Copy a batch of data from host to device
    vector<BatchData> data_vec;
    data_vec.clear();
    for (auto train : train_vec[net_id]) {
      auto batch = (*train)[itr];
      data_vec.push_back( batch );
    }

    dnnGraph_vec[net_id].feedForward(data_vec[0].x);
    dnnGraph_vec[net_id].backPropagate(data_vec, lr_vec[net_id], errorMeasure);

    ++itr_vec[net_id];
  }
  timer.elapsed();
  timer.reset();

  // output messages
  printf("Predicting...\n");
  timer.start();
  //cerr << "num_network: " << num_network << endl; //test
  for (int net_id=0; net_id<num_network; net_id++) {
    epoch = 0; //test

    vector<size_t> Ein = dnn_predict(dnnGraph_vec[net_id], train_vec[net_id], errorMeasure);
    vector<size_t> Eout = dnn_predict(dnnGraph_vec[net_id], valid_vec[net_id], errorMeasure);

    //cerr << "#outnet: " << Ein.size() << endl; //test
    for (int outnet_id=0; outnet_id<Ein.size(); outnet_id++) {
      printf("._______._________________________._________________________.\n"
             "|       |                         |                         |\n"
             "|       |        In-Sample        |      Out-of-Sample      |\n"
             "| Epoch |__________.______________|__________.______________|\n"
             "|       |          |              |          |              |\n"
             "|       | Accuracy | # of correct | Accuracy | # of correct |\n"
             "|_______|__________|______________|__________|______________|\n");
      fflush(stdout);

      float train_Acc = 1.0f - (float) Ein[outnet_id] / nTrain_vec[net_id];
    
      if (train_Acc < 0) {
          cout << "."; cout.flush();
          continue;
      }
    
      float valid_Acc = 1.0f - (float) Eout[outnet_id] / nValid_vec[net_id];
    
      printf("|%4lu   |  %5.2f %% |  %7lu     |  %5.2f %% |  %7lu     |\n",
        epoch, train_Acc * 100, nTrain_vec[net_id] - Ein[outnet_id], valid_Acc * 100, nValid_vec[net_id] - Eout[outnet_id]);
    }
    // Show Summary
    printf("\n%ld epochs in total\n", epoch);

    /*
    printf("[   In-Sample   ] ");
    showAccuracy(Ein, train_vec[i].size());
    printf("[ Out-of-Sample ] ");
    showAccuracy(Eout, valid_vec[i].size());
    */
  }
  timer.elapsed();
}

vector<size_t> dnn_predict(DNNGraph& dnnGraph, vector<DataSet*>& data_vec, ERROR_MEASURE errorMeasure) {
  vector<size_t> nError_vec( dnnGraph.getNOutput() );

  size_t nData = data_vec[0]->size();

  Batches batches(2048, nData);
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    // Copy a batch of data from host to device
    vector<BatchData> batch_vec;
    batch_vec.clear();
    for (auto data : data_vec) {
      auto batch = (*data)[itr];
      batch_vec.push_back( batch );
    }

    dnnGraph.feedForward(batch_vec[0].x);
    auto nError_vec_tmp = dnnGraph.countError(batch_vec, errorMeasure);
    for (size_t i=0; i<nError_vec.size(); i++)
        nError_vec[i] += nError_vec_tmp[i];
  }

  return nError_vec;
}

/*
size_t dnn_predict(const DNN& dnn, DataSet& data, ERROR_MEASURE errorMeasure) {
  size_t nError = 0;

  Batches batches(2048, data.size());
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    auto d = data[itr];
    mat prob = dnn.feedForward(d.x);
    nError += zeroOneError(prob, d.y, errorMeasure);
  }

  return nError;
}
*/

bool isEoutStopDecrease(const std::vector<size_t> Eout, size_t epoch, size_t nNonIncEpoch) {

  for (size_t i=0; i<nNonIncEpoch; ++i) {
    if (epoch - i > 0 && Eout[epoch] > Eout[epoch - i])
      return false;
  }

  return true;
}

