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

#include <dnn.h>

#include <iostream>  //TEST
using namespace std; //TEST

DNN::DNN(): _transforms(), _config(), _composite_net_flag(false) {} //Modified

DNN::DNN(string fn): _transforms(), _config(), _composite_net_flag(false) { //Modified
  this->read(fn);
}

DNN::DNN(const Config& config): _transforms(), _config(config), _composite_net_flag(false) { //Modified
}

DNN::DNN(const DNN& source): _transforms(source._transforms.size()), _config(), _composite_net_flag(false) { //Modified
  for (size_t i=0; i<_transforms.size(); ++i)
    _transforms[i] = source._transforms[i]->clone();
}

void DNN::init(const std::vector<mat>& weights) {
  throw std::runtime_error("\33[31m[Error]\33[0m Not implemented yet!!");
  /*_transforms.resize(weights.size());

  for (size_t i=0; i<_transforms.size() - 1; ++i)
      _transforms[i] = new Sigmoid(weights[i]);
  _transforms.back() = new Softmax(weights.back());*/
}

DNN::~DNN() { //Modified
  if( !_composite_net_flag )
    for (size_t i=0; i<_transforms.size(); ++i)
      delete _transforms[i];
}

DNN& DNN::operator = (DNN rhs) {
  swap(*this, rhs);
  return *this;
}
  
void DNN::setConfig(const Config& config) {
  _config = config;
}

size_t DNN::getNLayer() const {
  return _transforms.size() + 1;
}

size_t DNN::getNTransform() const {
  return _transforms.size();
}

void DNN::status() const {
  
  const auto& t = _transforms;

  size_t nAffines=0;
  for (size_t i=0; i<t.size(); ++i)
    nAffines += (t[i]->toString() == "AffineTransform");

  printf("\33[33m[INFO]\33[0m # of hidden layers: %2lu \n", nAffines - 1);

  for (size_t i=0; i<t.size(); ++i) {
    printf("  %-16s %4lu x %4lu [%-2lu]\n", t[i]->toString().c_str(),
        t[i]->getInputDimension(), t[i]->getOutputDimension(), i);
  }

}

void DNN::read(string fn) {

  FILE* fid = fopen(fn.c_str(), "r");

  if (!fid)
    throw std::runtime_error("\33[31m[Error]\33[0m Cannot load file: " + fn);

  _transforms.clear();

  FeatureTransform* f;
  while ( f = FeatureTransform::create(fid) )
    _transforms.push_back(f);

  fclose(fid);
}

void DNN::save(string fn) const {
  FILE* fid = fopen(fn.c_str(), "w");

  for (size_t i=0; i<_transforms.size(); ++i)
    _transforms[i]->write(fid);
  
  fclose(fid);
}

std::vector<FeatureTransform*>& DNN::getTransforms() {
  return _transforms;
}

const std::vector<FeatureTransform*>& DNN::getTransforms() const {
  return _transforms;
}

// ========================
// ===== Feed Forward =====
// ========================

void DNN::adjustLearningRate(float trainAcc) {
  static size_t phase = 0;

  if ( (trainAcc > 0.80 && phase == 0) ||
       (trainAcc > 0.85 && phase == 1) ||
       (trainAcc > 0.90 && phase == 2) ||
       (trainAcc > 0.92 && phase == 3) ||
       (trainAcc > 0.95 && phase == 4) ||
       (trainAcc > 0.97 && phase == 5)
     ) {

    float ratio = 0.9;
    printf("\33[33m[Info]\33[0m Adjust learning rate from \33[32m%.7f\33[0m to \33[32m%.7f\33[0m\n", _config.learningRate, _config.learningRate * ratio);
    _config.learningRate *= ratio;
    ++phase;
  }
}

mat DNN::feedForward(const mat& fin) const {
  mat y = feedForward_wo_resize(fin);
  y.resize(y.getRows(), y.getCols() - 1);
  return y;
}
mat DNN::feedForward_wo_resize(const mat& fin) const {

  mat y;

  _transforms[0]->feedForward(y, fin);

  for (size_t i=1; i<_transforms.size(); ++i)
    _transforms[i]->feedForward(y, y);

  //y.resize(y.getRows(), y.getCols() - 1);

  return y;
}

void DNN::feedForward(mat& output, const mat& fin) {
  feedForward_wo_resize(output, fin);
  output.resize(output.getRows(), output.getCols() - 1); //test
}

void DNN::feedForward_wo_resize(mat& output, const mat& fin) {

  // FIXME This should be an ASSERTION, not resizing.
  if (_houts.size() != this->getNLayer() - 2)
    _houts.resize(this->getNLayer() - 2);

  if (_houts.size() > 0) {
    _transforms[0]->feedForward(_houts[0], fin);

    for (size_t i=1; i<_transforms.size()-1; ++i)
      _transforms[i]->feedForward(_houts[i], _houts[i-1]);

    _transforms.back()->feedForward(output, _houts.back());
  }
  else {
    _transforms.back()->feedForward(output, fin);
  }

  //output.resize(output.getRows(), output.getCols() - 1); //test
}

// ============================
// ===== Back Propagation =====
// ============================

void DNN::backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate, const vector<bool>& layerOn) {
  mat output(fout);
  output.reserve(output.size() + output.getRows());
  output.resize(output.getRows(), output.getCols() + 1);

  error.reserve(error.size() + error.getRows());
  error.resize(error.getRows(), error.getCols() + 1);

  backPropagate_wo_resize(error, fin, fout, learning_rate, layerOn);
}

void DNN::backPropagate_wo_resize(mat& error, const mat& fin, const mat& fout, float learning_rate, const vector<bool>& layerOn) {
//void DNN::backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) {

  assert(error.getRows() == fout.getRows() && error.getCols() == fout.getCols());

  if (_houts.size() > 0) {
    _transforms.back()->backPropagate(error, _houts.back(), fout, (layerOn.back() ? learning_rate : 0));
    //_transforms.back()->backPropagate(error, _houts.back(), fout, learning_rate);

    for (int i=_transforms.size() - 2; i >= 1; --i)
      _transforms[i]->backPropagate(error, _houts[i-1], _houts[i], (layerOn[i] ? learning_rate : 0));
      //_transforms[i]->backPropagate(error, _houts[i-1], _houts[i], learning_rate);

    _transforms[0]->backPropagate(error, fin, _houts[0], (layerOn[0] ? learning_rate : 0));
    //_transforms[0]->backPropagate(error, fin, _houts[0], learning_rate);
  }
  else
    _transforms.back()->backPropagate(error, fin, fout, (layerOn.back() ? learning_rate : 0));
    //_transforms.back()->backPropagate(error, fin, fout, learning_rate);
}

void DNN::backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) { // For back-compatibility with cnn-train.cpp
  mat output(fout);
  output.reserve(output.size() + output.getRows());
  output.resize(output.getRows(), output.getCols() + 1);

  error.reserve(error.size() + error.getRows());
  error.resize(error.getRows(), error.getCols() + 1);

  backPropagate_wo_resize(error, fin, fout, learning_rate);
}

void DNN::backPropagate_wo_resize(mat& error, const mat& fin, const mat& fout, float learning_rate) { // For back-compatibility with cnn-train.cpp

  assert(error.getRows() == fout.getRows() && error.getCols() == fout.getCols());

  if (_houts.size() > 0) {
    _transforms.back()->backPropagate(error, _houts.back(), fout, learning_rate);

    for (int i=_transforms.size() - 2; i >= 1; --i)
      _transforms[i]->backPropagate(error, _houts[i-1], _houts[i], learning_rate);

    _transforms[0]->backPropagate(error, fin, _houts[0], learning_rate);
  }
  else
    _transforms.back()->backPropagate(error, fin, fout, learning_rate);
}

void DNN::backPropagate_fdlr(mat& error, const mat& fin, const mat& fout, float learning_rate) { //New

  mat output(fout);
  output.reserve(output.size() + output.getRows());
  output.resize(output.getRows(), output.getCols() + 1);

  error.reserve(error.size() + error.getRows());
  error.resize(error.getRows(), error.getCols() + 1);

  assert(error.getRows() == output.getRows() && error.getCols() == output.getCols());

  if (_houts.size() > 0) {
    _transforms.back()->backPropagate(error, _houts.back(), output, 0);

    for (int i=_transforms.size() - 2; i >= 1; --i)
      _transforms[i]->backPropagate(error, _houts[i-1], _houts[i], 0);

    _transforms[0]->backPropagate(error, fin, _houts[0], learning_rate);
  }
  else
    _transforms.back()->backPropagate(error, fin, output, learning_rate);
}

//void DNN::backPropagate_top2_bottom1(mat& error, const mat& fin, const mat& fout, float learning_rate) { //New
void DNN::backPropagate_top_bottom(mat& error, const mat& fin, const mat& fout, float learning_rate, int num_adapt_top, int num_adapt_bottom) { //New
  //TODO: implement num_adapt_bottom

  mat output(fout);
  output.reserve(output.size() + output.getRows());
  output.resize(output.getRows(), output.getCols() + 1);

  error.reserve(error.size() + error.getRows());
  error.resize(error.getRows(), error.getCols() + 1);

  assert(error.getRows() == output.getRows() && error.getCols() == output.getCols());

  int cur_layer_from_top = 0;
  //int num_adapt = 2; //test
  //int num_adapt = 1; //test
  int num_adapt = num_adapt_top; //test
  bool update = true;
  if (_houts.size() > 0) {
    _transforms.back()->backPropagate(error, _houts.back(), output, 0);
    //cout << _transforms.back()->toString() << endl; //TEST

    for (int i=_transforms.size() - 2; i >= 1; --i) {
      if (update && _transforms[i]->toString() == "AffineTransform") {
        //cout << _transforms[i]->toString() << endl; //TEST
        _transforms[i]->backPropagate(error, _houts[i-1], _houts[i], learning_rate);
        cur_layer_from_top++;
        if (cur_layer_from_top >= num_adapt)
          update = false;
      }
      else
        _transforms[i]->backPropagate(error, _houts[i-1], _houts[i], 0);
    }

    _transforms[0]->backPropagate(error, fin, _houts[0], learning_rate);
  }
  else
    _transforms.back()->backPropagate(error, fin, output, learning_rate);
}

const Config DNN::getConfig() const {
  return _config;
}

Config& DNN::getConfig() {
  return _config;
}

void swap(DNN& lhs, DNN& rhs) {
  using std::swap;
  swap(lhs._transforms, rhs._transforms);
  swap(lhs._config, rhs._config);
}

void DNN::append(DNN& subnet) { //New method
  _composite_net_flag = true;

  for (size_t i=0; i<subnet._transforms.size(); ++i)
    _transforms.push_back( subnet._transforms[i] );
}

void DNN::composite(DNN& full_net, DNN& input_net, DNN& hidden_net, DNN& out_net, vector<size_t> layer_spec_vec) { //New method
  layer_spec_vec.clear();
  layer_spec_vec.push_back( input_net._transforms.size() );
  layer_spec_vec.push_back( hidden_net._transforms.size() );
  layer_spec_vec.push_back( out_net._transforms.size() );

  full_net._composite_net_flag = true;
  full_net._transforms.clear();

  for (size_t i=0; i<input_net._transforms.size(); ++i)
    full_net._transforms.push_back( input_net._transforms[i] );
  for (size_t i=0; i<hidden_net._transforms.size(); ++i)
    full_net._transforms.push_back( hidden_net._transforms[i] );
  for (size_t i=0; i<out_net._transforms.size(); ++i)
    full_net._transforms.push_back( out_net._transforms[i] );
}

void DNN::split(DNN& full_net, DNN& input_net, DNN& hidden_net, DNN& out_net, vector<size_t> layer_spec_vec) { //New method
  size_t base = 0;
  for (size_t i=0; i<layer_spec_vec[0]; ++i)
    input_net._transforms.push_back( full_net._transforms[i] );
  base += layer_spec_vec[0];
  for (size_t i=0; i<layer_spec_vec[1]; ++i)
    hidden_net._transforms.push_back( full_net._transforms[i] );
  base += layer_spec_vec[1];
  for (size_t i=0; i<layer_spec_vec[2]; ++i)
    out_net._transforms.push_back( full_net._transforms[i] );
}

// =============================
// ===== Utility Functions =====
// =============================

/*mat l2error(mat& targets, mat& predicts) {
  mat err(targets - predicts);

  thrust::device_ptr<float> ptr(err.getData());
  thrust::transform(ptr, ptr + err.size(), ptr, func::square<float>());

  mat sum_matrix(err.getCols(), 1);
  err *= sum_matrix;
  
  return err;
}
*/
