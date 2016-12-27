#ifndef __DNN_GRAPH_H_
#define __DNN_GRAPH_H_

#include <vector>
#include <string>
#include <dnn.h>

#define NULL_NET nullptr

class Node {
public:
  Node() {}

  Node(string name, DNN* net);
  //Node(string name, DNN* net, Node* in, Node* out): _seen(false), _name(name), _net(net) {
  Node(string name, DNN* net, Node* in);

  Node(const Node& source);

  void feedForward(const mat& fin);

  void backPropagate_lastLayer(mat& error, float learning_rate);
  void backPropagate(const mat& error, float learning_rate);
  //void backPropagate(mat& error, float learning_rate);

  size_t getInputDimension() const { return _net->getTransforms().front()->getInputDimension(); }
  size_t getOutputDimension() const { return _net->getTransforms().back()->getOutputDimension(); }

  bool          _seen;
  string        _name;
  DNN*          _net;
  vector<Node*> _fanin,
                _fanout;
  mat           _fin;
  mat           _hout;
  mat           _error;

  float         _objectiveWeight;
};

class DNNGraph {
public:
  DNNGraph();

  void addPath(vector<DNN*>& subnets, vector<string>& names, vector<bool>& layerOn, float objectiveWeight);
  void feedForward(const mat& fin);
  void backPropagate(vector<BatchData>& data_vec, float learning_rate, ERROR_MEASURE errorMeasure);
  vector<size_t> countError(vector<BatchData>& data_vec, ERROR_MEASURE errorMeasure);

  //vector<mat*> getOutputs();
  size_t getNOutput() { return _node_out._fanin.size(); }

  void setConfig(const Config& config) { _config = config; }
  Config getConfig() const { return _config; }
  Config& getConfig() { return _config; }

private:
  Node              _node_in;
  Node              _node_out;
  map<string, Node> _nodes;

  Config _config;
};

#endif //end dnnGraph.h
