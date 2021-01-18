#ifndef KNNDup_HPP_
#define KNNDup_HPP_


#include <vector>
#include <random>
#include <iostream>

//#include <utility>
//#include <functional>
//#include <string>
//#include <cmath>
//#include <algorithm>

#include "../matTools.hpp"

#include <Eigen/Dense>
#include <knn/kdtree_minkowski.h>



using Mat       = Eigen::MatrixXd;
using Vect      = Eigen::VectorXd;


class KNNDup{
public:

	//! constructors
	KNNDup( );// { }
	KNNDup( Mat, double , int k_ = 3 );


	void makeNewPoints();
	void CombineNewX();

	//! destructor
	~KNNDup( ) { }

	Mat colWisePoints;
	Mat newPoints;
	Mat combinedNewX;


	double propNewP;
	int k;


private:

	void removeDuplicateCols( Mat& );

	Eigen::VectorXi knnPointIndex;

};


KNNDup::KNNDup( Mat rowWisePoints, double propNewP_, int k_){

	colWisePoints = rowWisePoints.transpose();
	k = k_;
	propNewP = propNewP_;

}


void KNNDup::makeNewPoints(){

	int numNewP = propNewP * colWisePoints.cols();

	std::cout << "propNewP = " << propNewP << std::endl;
	std::cout << "colWisePoints.cols() = " <<  colWisePoints.cols() << std::endl;
	std::cout << "numNewP = " << numNewP << std::endl;

	std::random_device rd;
	std::mt19937 engine( rd() );
	std::uniform_int_distribution<int> uniform(0, colWisePoints.cols() - 1 );


	//can find better ways to uniquely sample existing points for knn
	knnPointIndex.resize(numNewP);

	std::vector <int> uniqueIntIndex( colWisePoints.cols() );

	for(int i = 0; i < colWisePoints.cols() ; ++i){
		uniqueIntIndex[i] = i;
	}

	std::shuffle(uniqueIntIndex.begin(), uniqueIntIndex.end(), std::default_random_engine(42) );

	for(int i = 0; i < knnPointIndex.size() ; ++i){
		knnPointIndex[i] = uniqueIntIndex[i];
	}

	//std::cout << "knnPointIndex\n" << knnPointIndex << std::endl;

    Mat queryPoints( colWisePoints.rows(), knnPointIndex.size() );

    for(int i = 0; i < knnPointIndex.size(); ++i){

    	queryPoints.col(i) = colWisePoints.col(knnPointIndex[i]);

    }


	/*
		https://github.com/Rookfighter/knn-cpp

		Songrit Maneewongvatana and David M. Mount,
		Analysis of Approximate Nearest Neighbor Searching with Clustered Point Sets,
		DIMACS Series in Discrete Mathematics and Theoretical Computer Science, 2002

		Mohammad Norouzi, Ali Punjani and David J. Fleet,
		Fast Search in Hamming Space with Multi-Index Hashing,
		In Proceedings of 2012 IEEE Conference on Computer Vision and Pattern Recognition
	*/

	knn::KDTreeMinkowski<double, knn::EuclideanDistance<double> > kdtree( colWisePoints );
	kdtree.setBucketSize(16);
    kdtree.setCompact(false);
    kdtree.setBalanced(false);
    kdtree.setSorted(true);
    kdtree.setTakeRoot(true);
    kdtree.setMaxDistance(0);
    kdtree.setThreads(-1);

    kdtree.build();

	knn::Matrixi indices;
    Mat distances;

    kdtree.query(queryPoints, k, indices, distances);

    newPoints.resize( colWisePoints.rows(), knnPointIndex.size() );
    Mat temp(colWisePoints.rows(), k );

    for(int i = 0; i < knnPointIndex.size(); ++ i){

    	for(int j = 0; j < k; j++){
    		temp.col(j) = colWisePoints.col(indices(j,i) );
    	}

    	newPoints.col(i) = temp.rowwise().mean();
    }


}

void KNNDup::CombineNewX(){

	combinedNewX.resize(colWisePoints.rows(), colWisePoints.cols() + newPoints.cols() );

	//combinedNewX.block(0, 0, colWisePoints.rows(), colWisePoints.cols()) = colWisePoints;

	//combinedNewX.block(0, colWisePoints.cols(), newPoints.rows(), newPoints.cols() ) = newPoints;

	combinedNewX << colWisePoints, newPoints;

	combinedNewX.transposeInPlace();

	matTools::removeDuplicateRows(combinedNewX);

}




#endif // KNNDup_HPP_
