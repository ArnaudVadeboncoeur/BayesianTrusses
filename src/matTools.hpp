


#ifndef matTools_HPP_
#define matTools_HPP_

#include <iostream>
#include <Eigen/Dense>

using Mat = Eigen::MatrixXd;

class matTools{

	public:

		static void removeRow(Mat& matrix, unsigned int rowToRemove){

			//Function to remove a row from Eigen Matrix by reference
			unsigned int numRows = matrix.rows()-1;
			unsigned int numCols = matrix.cols();

			if( rowToRemove < numRows ){
				matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);
			}
			matrix.conservativeResize(numRows,numCols);

		}


		static void removeColumn(Mat& matrix, unsigned int colToRemove){
			//Function to remove a column from Eigen Matrix by reference
			unsigned int numRows = matrix.rows();
			unsigned int numCols = matrix.cols()-1;

			if( colToRemove < numCols ){
				matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);
			}
			matrix.conservativeResize(numRows,numCols);
		}



		static void removeDuplicateRows(Mat& A){
			//remove duplicate rows from final row major total new X Matrix

			for(int i = A.rows() - 1 ; i >= 0 ; i--){

				for(int j = i - 1; j >=0; j--){

					if(A.row(i).isApprox(A.row(j), 1e-9)){

						matTools::removeRow(A, i);
						break;

					}
				}
				}
		}

};

#endif /* matTools_HPP_ */
