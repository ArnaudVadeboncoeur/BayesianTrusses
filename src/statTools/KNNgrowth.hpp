#ifndef KNNDup_HPP_
#define KNNDup_HPP_

#include <utility>
#include <functional>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>
#include <knn/kdtree_minkowski.h>



using Mat       = Eigen::MatrixXd;
using Vect      = Eigen::VectorXd;


class KNNDup{
public:

	//! constructors
	KNNDup( );// { }
	KNNDup( Mat, int, int kIn = 2 );

	void makeNewPoints();

	//! destructor
	~KNNDup( ) { }

	Mat ColWisePoints;
	Mat newPoints;


	int numNewP;
	int k;


private:

};

KNNDup::KNNDup( Mat rowWisePoints, int numNewP_, int k_){

	ColWisePoints = rowWisePoints.transpose();
	k = k_;
	numNewP = numNewP_;

}


void KNNDup::makeNewPoints(){

    Mat dataPoints(3, 9);
    dataPoints << 1, 2, 3, 1, 2, 3, 1, 2, 3,
                  2, 1, 0, 3, 2, 1, 0, 3, 4,
                  3, 1, 3, 1, 3, 4, 4, 2, 1;

	knn::KDTreeMinkowski<double, knn::EuclideanDistance<double> > kdtree( dataPoints );
	kdtree.setBucketSize(16);
    kdtree.setCompact(false);
    kdtree.setBalanced(false);
    kdtree.setSorted(true);
    kdtree.setTakeRoot(true);
    kdtree.setMaxDistance(0);
    kdtree.setThreads(-1);

    kdtree.build();

    Mat queryPoints(3, 1);
	//queryPoints << 0, 1, 0;
    queryPoints = dataPoints.block(0,0, 3, 3);

	knn::Matrixi indices;
    Mat distances;

    kdtree.query(queryPoints, 3, indices, distances);

    std::cout
	<< "Data points:" << std::endl
	<< dataPoints << std::endl
	<< "Query points:" << std::endl
	<< queryPoints << std::endl
	<< "Neighbor indices:" << std::endl
	<< indices << std::endl
	<< "Neighbor distances:" << std::endl
	<< distances << std::endl;


}






#endif // KNNDup_HPP_
