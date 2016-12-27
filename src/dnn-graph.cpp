#include <dnn-graph.h>
//******************************************************************************************************
//                 Class  Node
//******************************************************************************************************


Node::Node(string name, DNN* net): _seen(false), _name(name), _net(net), _objectiveWeight(1) {
}

//Node(string name, DNN* net, Node* in, Node* out): _seen(false), _name(name), _net(net), _objectiveWeight(1) {
Node::Node(string name, DNN* net, Node* in): _seen(false), _name(name), _net(net), _objectiveWeight(1) {
    _fanin.push_back(in);
    //_fanout.push_back(out);
}

Node::Node(const Node& source) {
    _seen   = false;
    _name   = source._name;
    _net    = source._net;
    _fanin  = source._fanin;
    _fanout = source._fanout;
}

void Node::feedForward(const mat& fin) { //traverse order: all fanin -> self -> fanouts
    //cerr << _name << endl; //test
    if (_net == nullptr) {
    //if (_name == "NODE_OUT") {
        //cerr << "reached NODE_OUT" << endl; //test
        return;
    }

    if (!_seen)
        _fin = fin;
    else
        _fin += fin;
    /*
    cerr << "_seen " << (_seen?"true":"false") << endl; //test
    cerr << "_fin" << endl; //test
    hmat(_fin).info(); //test
    fflush(stdout); //test
    cerr << "fin" << endl; //test
    hmat(fin).info(); //test
    fflush(stdout); //test
    */

    _seen = true;

    bool precede = true;
    if (_fanin.size() > 1)
        for(auto fanin : _fanin)
            precede = precede && fanin->_seen;
    if (precede) {
        /*
        //_net->feedForward(_hout, _fin);
        hmat(_fin).info(); //test
        fflush(stdout); //test
        */
        //cerr << _name << endl; //test
        _net->feedForward_wo_resize(_hout, _fin); //test

        for(auto fanout : _fanout)
            fanout->feedForward(_hout);
    }
}

void Node::backPropagate_lastLayer(mat& error, float learning_rate) {
    size_t size = _hout.size();
    size_t nRow = _hout.getRows();
    size_t nCol = _hout.getCols();
    _hout.reserve(size + nRow); //test ???
    _hout.resize(nRow, nCol + 1);

    error.reserve(error.size() + error.getRows()); //test ???
    error.resize(error.getRows(), error.getCols() + 1);

    backPropagate(error, learning_rate);
}

void Node::backPropagate(const mat& error, float learning_rate) {
    //cerr << _name << endl; //test
    if (_net == nullptr) {
    //if (_name == "NODE_IN") {
        //cerr << "reached NODE_IN!" << endl; //test
        return;
    }

    if (!_seen)
        _error = error;
    else
        _error += error;
    /*
    cerr << "_error" << endl; //test
    hmat(_error).info(); //test
    cerr << "error" << endl; //test
    hmat(error).info(); //test
    fflush(stdout); //test
    */

    _seen = true;

    bool precede = true;
    if (_fanout.size() > 1)
        for(auto fanout : _fanout)
            precede = precede && fanout->_seen;
    if (precede) {
        /*
        hmat(error).info(); //test
        hmat(_fin).info(); //test
        hmat(_hout).info(); //test
        */
        auto layerOn = _net->getConfig().layerOn;
        _net->backPropagate_wo_resize(_error, _fin, _hout, learning_rate, layerOn);

        for(auto fanin : _fanin)
            fanin->backPropagate(_error, learning_rate);
    }
}





//******************************************************************************************************
//                 Class  DNNGraph
//******************************************************************************************************

DNNGraph::DNNGraph() : _node_in("NODE_IN", NULL_NET), _node_out("NODE_OUT", NULL_NET) {
    /*
    _node_in._fanin.push_back(nullptr); //test
    _node_out._fanout.push_back(nullptr); //test
    */
}

void DNNGraph::addPath(vector<DNN*>& subnets, vector<string>& names, vector<bool>& layerOn, float objectiveWeight) {
    assert(subnets.size() > 0);
    size_t beg=0, end=0;
    for (auto net : subnets) {
        end += net->getNTransform();
        net->getConfig().layerOn = vector<bool> (layerOn.begin()+beg, layerOn.begin()+end);
        beg = end;
    }

    Node* prevNode = & _node_in;
    Node* curNode;
    //cout << "path: "; //test
    for(size_t i=0; i<subnets.size(); i++) {
        //cerr << names[i] << " "; //test
        if( _nodes.find(names[i]) == _nodes.end() ) {
            _nodes.insert( pair<string, Node>(names[i], Node(names[i], subnets[i], prevNode)) );
        }
        curNode = &( _nodes[names[i]] );
        prevNode->_fanout.push_back( curNode );
        prevNode = curNode;
        //cerr << "#nodes: " << _nodes.size() << endl; //test
    }
    //cout << endl; //test

    curNode->_objectiveWeight = objectiveWeight;

    bool existed = false;
    for (auto node : _node_out._fanin)
        if (node == prevNode) {
            existed = true;
            break;
        }
    if(!existed) {
        prevNode->_fanout.push_back( &_node_out );
        _node_out._fanin.push_back( prevNode );
    }
    cerr << "#outnet: " << _node_out._fanin.size() << endl;; //test
    //cerr << _node_out._fanin[0]->_name << endl;; //test
}

void DNNGraph::feedForward(const mat& fin) {
    //cerr << "feedForward" << endl; //test
    //for(auto itr=_nodes.begin(); itr!=_nodes.end(); ++itr) {
    for(auto& elem : _nodes) {
        //Node& node = (*itr).second;
        Node& node = elem.second;
        node._seen = false;
    }
    _node_in._seen = true;

    vector<Node*>& net_in = _node_in._fanout;
    for(auto net : net_in) {
        net->feedForward(fin);
    }

    vector<Node*>& net_out = _node_out._fanin;
    for(auto net : net_out) {
        size_t nRow = net->_hout.getRows();
        size_t nCol = net->_hout.getCols() - 1;
        net->_hout.resize(nRow, nCol);
    }
}

void DNNGraph::backPropagate(vector<BatchData>& data_vec, float learning_rate, ERROR_MEASURE errorMeasure) {
    //cerr << "backPropagate" << endl; //test
    //for(auto itr=_nodes.begin(); itr!=_nodes.end(); ++itr) {
    for(auto& elem : _nodes) {
        //Node& node = (*itr).second;
        Node& node = elem.second;
        node._seen = false;
    }
    _node_out._seen = true;

    vector<Node*>& net_out = _node_out._fanin;

    for(size_t i=0; i<net_out.size(); i++) {
        //data_vec[i].y.info(); //test
        //hmat(net_out[i]->_hout).info(); //test
        mat error = getError( data_vec[i].y, net_out[i]->_hout, errorMeasure ) * (net_out[i]->_objectiveWeight);

        net_out[i]->backPropagate_lastLayer(error, learning_rate);
    }
}

vector<size_t> DNNGraph::countError(vector<BatchData>& data_vec, ERROR_MEASURE errorMeasure) {
    vector<size_t> nError_vec;

    vector<Node*>& net_out = _node_out._fanin;

    //cerr << "#outnet: " << _node_out._fanin.size() << endl;; //test
    for(size_t i=0; i<net_out.size(); i++)
        nError_vec.push_back( zeroOneError( net_out[i]->_hout, data_vec[i].y, errorMeasure ) );

    //cerr << "#outnet: " << nError_vec.size() << endl;; //test
    return nError_vec;
}
