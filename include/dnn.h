#ifndef __DNN_H_
#define __DNN_H_

#include <dnn-utility.h>
#include <dataset.h>
#include <feature-transform.h>
#include <config.h>
#include <utility.h>

class DNN {
public:
  DNN();
  DNN(string fn);
  DNN(const Config& config);
  DNN(const DNN& source);
  ~DNN();

  DNN& operator = (DNN rhs);

  void init(const std::vector<mat>& weights);

  mat feedForward(const mat& fin) const;
  mat feedForward_wo_resize(const mat& fin) const; //test
  void feedForward(mat& output, const mat& fin);
  void feedForward_wo_resize(mat& output, const mat& fin); //test
  void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate, const vector<bool>& layerOn);
  void backPropagate_wo_resize(mat& error, const mat& fin, const mat& fout, float learning_rate, const vector<bool>& layerOn);
  void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate); // For back-compatibility with cnn-train.cpp
  void backPropagate_wo_resize(mat& error, const mat& fin, const mat& fout, float learning_rate); // For back-compatibility with cnn-train.cpp
  void backPropagate_fdlr(mat& error, const mat& fin, const mat& fout, float learning_rate); //New
  void backPropagate_top_bottom(mat& error, const mat& fin, const mat& fout, float learning_rate, int num_adapt_top, int num_adapt_bottom); //New
  //void backPropagate_top2_bottom1(mat& error, const mat& fin, const mat& fout, float learning_rate); //New
  //void backPropagate_top1_bottom1(mat& error, const mat& fin, const mat& fout, float learning_rate); //New

  void setConfig(const Config& config);
  size_t getNLayer() const;
  size_t getNTransform() const;

  const Config getConfig() const;
  Config& getConfig();
  void adjustLearningRate(float trainAcc);

  void status() const;

  void read(string fn);
  void save(string fn) const;

  void append(DNN& subnet); //New method

  static void composite(DNN& full_net, DNN& input_net, DNN& hidden_net, DNN& out_net, vector<size_t> layer_spec_vec); //New method
  static void split(DNN& full_net, DNN& input_net, DNN& hidden_net, DNN& out_net, vector<size_t> layer_spec_vec); //New method

  std::vector<FeatureTransform*>& getTransforms();
  const std::vector<FeatureTransform*>& getTransforms() const;

  friend void swap(DNN& lhs, DNN& rhs);

private:
  std::vector<FeatureTransform*> _transforms;

  /* Hidden Outputs: outputs of each hidden layers
   * The first element in the std::vector (i.e. _houts[0])
   * is the output of first hidden layer. 
   * ( Note: this means no input data will be kept in _houts. )
   * ( Also, no output data will be kept in _houts. )
   * */
  std::vector<mat> _houts;
  Config _config;

  bool _composite_net_flag; //New
};

void swap(DNN& lhs, DNN& rhs);
#endif  // __DNN_H_
