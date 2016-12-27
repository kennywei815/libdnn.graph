#ifndef __BATCH_MOD_H_
#define __BATCH_MOD_H_
//deprecated

#include <string>
#include <batch.h>
#include <dataset.h>

class Batches_mod : public Batches {
public:
  //Batches_mod(size_t batchSize, size_t totalSize) : Batches(batchSize, totalSize){}
  /*
  Batches_mod(size_t batchSize, size_t totalSize, DataSet* dataSetPtr) :
    Batches(batchSize, totalSize), _dataSetPtr(dataSetPtr),
    _begin(dataSetPtr, Batches::begin()), _end(dataSetPtr, Batches::end()) {}
  */

  Batches_mod(size_t batchSize, size_t totalSize, string name, DataSet* dataSetPtr) :
    Batches(batchSize, totalSize),
    _name(name), _dataSetPtr(dataSetPtr),
    _begin(dataSetPtr, Batches::begin()), _end(dataSetPtr, Batches::end()) {}

  friend class iterator;
  class iterator : public Batches::iterator {
  public:
    friend class Batches_mod;

    iterator(const iterator& source): Batches::iterator(source), _dataSetPtr(source._dataSetPtr) {}

    //iterator& operator = (iterator rhs);
    //iterator& operator = (Batches::iterator rhs); //don't want
    
    //const Batch& operator * () const { //TODO
    BatchData operator * () { //TODO
      //assert(_dataSetPtr != null_ptr); //TODO
      /*
      Batches::iterator* p   = this;
      Batches::iterator  itr = *p;
      return (*_dataSetPtr)[itr];
      */
      //return _dataSetPtr->operator [] (*dynamic_cast<Batches::iterator*>(this));
      //return (*_dataSetPtr)[*dynamic_cast<Batches::iterator*>(this)];
      return (*_dataSetPtr)[*static_cast<Batches::iterator*>(this)];
      //return p->operator * (); //fall back
    }

  private:
    DataSet* _dataSetPtr;

    iterator(DataSet* dataSetPtr, const Batches::iterator& source) : Batches::iterator(source), _dataSetPtr(dataSetPtr) {}
    //iterator(DataSet* dataSetPtr, int index, size_t batchSize, size_t totalSize): iterator(index, batchSize, totalSize), _dataSetPtr(dataSetPtr)
    //iterator(DataSet* dataSetPtr, int index, size_t batchSize, size_t totalSize): iterator(index, batchSize, totalSize), _dataSetPtr(dataSetPtr)
    //iterator(const iterator& source): Batches::iterator(source) {}
  };

  const iterator& begin() const { return _begin; }
  const iterator& end() const { return _end; }
private:
  string   _name;
  DataSet* _dataSetPtr;

  iterator _begin;
  iterator _end;
};
#endif // __BATCH_MOD_H_
