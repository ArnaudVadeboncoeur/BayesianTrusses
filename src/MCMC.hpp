#ifndef MCMC_HPP_
#define MCMC_HPP_

#include <utility>
#include <functional>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>



template< unsigned DIM, typename FUNC, typename ARG >
class MCMC{
public:
    //! public vars
    //  user choice of sampling methods
    enum METHOD {METRO, OTHER};
    METHOD method;

	//! constructors
	//error message stops after this formulation..
	MCMC( );// { }

	MCMC( FUNC func, const ARG & xInitial,ARG sigmaInitial,
	      const std::pair< ARG, ARG > & bounds,
	      const unsigned,
	      const METHOD,
	      const bool,
	      const bool classVerbosity = false );

	//! deal with the choice of simga
	//  find optimal sigmaStep based on a one side bound line search method
	void setSigma( ARG simgaJump, const unsigned maxStep ,
				   unsigned optSampleSize, const bool setSigmaVerbose = false );

	//! if Optimal sigma is know, set it
	void setSigma( ARG knownSigma ){ sigmaStep_ = knownSigma; }

	//! function to creat specified number of randomly sampled points
	void sample( unsigned nSamlpes );

	//! queries
	ARG getMaxArg( ) { return xMax_;}
	double getMaxVal( ) { return valMax_;}
	unsigned getAccepted( ) { return acceptedCandidates_.size();}
	void writeResult( std::ostream & );


	//! return accepted candidates vector of pairs
	std::vector< std::pair< ARG, double > >	getAcceptedCandidates( ) {return acceptedCandidates_;}

	//return all sigma jumps of optimisation process
	std::vector< std::pair < ARG, double > > getAllSigmaJumps( ) {return allSigmaJumps_;}

	//! destructor
	~MCMC( ) { }

private:
	//! Proposal distribution for xProposed
	ARG transitionKernel_();

	//! user-defined parameters
	FUNC func_;
	ARG sigmaStep_;
	ARG x_;
	ARG xMax_;
	std::pair< ARG, ARG > argBounds_;
	unsigned burnInNum_;
	double acceptRatio_;
	double valMax_;

	bool classVerbosity_;
	bool uniformSigma_;

	//! containers
	std::vector< std::pair < ARG, double > > allSigmaJumps_;
	std::vector< std::pair<ARG, double> >	acceptedCandidates_;

	//! MCMC process
	unsigned rejectedNum_;
	double acceptanceRatio_;
	std::vector<double> maxArg_;

    std::random_device rd_;
    std::mt19937 engine_ ;
};



template<unsigned DIM, typename FUNC, typename ARG>
MCMC<DIM,FUNC,ARG>::MCMC
/*
 *
 * \parm func:     function returning double, pdf to be sampled
 *       xInitial:       vector,             starting position of x_
 *       sigmaInitial:   vector              starting value of sigma for optimisation, recommended (DIM, 0.01)
 *       bounds:         pair of vectors,    of form std::make_pair(Vec lower, Vec upper)
 *       burnInNUm:      int,                number of samples to be dropped from beginning, recommended 100
 *       algMethod:      enum METHOD         choice of the algorithm, METRO, OTHER that is not specified yet
 *       uniformSigma:   bool,               whether or not to use different sigma scaling, is true set sigmaStep to
 *                                           be proportional to step size evolution
 *       classVerbosity_ bool,               Whether or not to print regular messages on progress of methods
 */
    ( FUNC func, const ARG & xInitial,ARG sigmaInitial,
      const std::pair< ARG, ARG > & bounds,
      const unsigned burnInNum,
      const METHOD algMethod,
      const bool uniformSigma,
      const bool classVerbosity ){

	//class constructor to initialise most needed parameters

	//Question: How can you pass "enum METHOD" as constructor argument?//

	// initialise parameters
	burnInNum_        = burnInNum;
	func_             = func;
	argBounds_        = bounds;
	sigmaStep_        = sigmaInitial;
	method            = algMethod;
	classVerbosity_   = classVerbosity;
	uniformSigma_     = uniformSigma;

	//! initialise the process parameters with 0.0
	acceptRatio_      = 0.0;
	valMax_           = 0.0;
	rejectedNum_      = 0;
	acceptanceRatio_  = 0.0;

	//! set initial argument
	x_ 				  = xInitial;

    std::random_device rd_;
    std::mt19937 engine_ ;

	acceptedCandidates_.reserve( burnInNum );
	allSigmaJumps_.reserve( 1 );
}


template<unsigned DIM, typename FUNC, typename ARG>
ARG MCMC<DIM,FUNC,ARG>::transitionKernel_(){

	//private method give next proposal vector xProp that will be tested in methode sample

	ARG xProp(DIM);

	/*
	 * If uniformSigma_ is true, then a shortcut can be taken as seen bellow and only creat one instance of Normal
	 * with mean 0. Normal is the added to the current x_ acting as if it were centred at x_ .
	 */
	if( uniformSigma_ ){

		std::normal_distribution<double> Normal( 0, sigmaStep_[0] );
		std::random_device rd;
		std::mt19937 engine( rd() );
		//std::mt19937 engine( 10 );
		//engine_ (rd_);

		//std::mt19937 engine( 2 );
		//std::default_random_engine engine;

		for( unsigned int i = 0; i < DIM ; i++ ){

			xProp[i] = x_[i] + Normal( engine );
		}
		return xProp;

	//uniformSigma_ is not true, then one must create a new instance of Normal with the new sigma vector
	}else{
		for( unsigned int i =0; i < DIM; ++i ){

			std::normal_distribution<double> normal( x_[i], sigmaStep_[i] );
			std::random_device rd;
			std::mt19937 engine( rd() );

			xProp[i] = normal( engine );
		}
	}
	return xProp;
}


template<unsigned DIM, typename FUNC, typename ARG>
void MCMC<DIM,FUNC,ARG>::sample( unsigned nSamples ){

    if( classVerbosity_ ){

        std::cout << "\n\n";
        std::cout << "sigmaStep_ = : \n"<< std::endl << "\t";

        for( int i =0; i < DIM; ++i){

            std::cout << sigmaStep_[i] << " ";
        }

        std::cout << '\n';
     }

	//methode to creat samples from the pdf function.
	//currently only support metropolis.

	acceptanceRatio_ = 0;
	valMax_			 = 0;

	std::vector< std::pair<ARG, double> > emptyAcceptedCandidates ( 0 );
	acceptedCandidates_ 				= emptyAcceptedCandidates;

	acceptedCandidates_.reserve(0.25 * nSamples);

	//This is the uniform distribution needed to accept next sample with probability r.
	std::uniform_real_distribution<double> uniformDistribution( 0, 1 );
	std::random_device rd;
	std::mt19937 engine( rd() );

	unsigned totalPoints = 0;

	double Ratio, randNum, yCurr, yProp;

	ARG xProp( DIM );

	//defining max points as a very low acceptance ratio of 5%
	const unsigned maxPoints = ceil( nSamples * 1.0 / 0.05 );

	switch( method ){

	case METRO:

	        //Could try and implement a student-t proposal distribution for bridging domain gaps

			if( classVerbosity_ ) { std::cout << "Entering sample Metropolis loop" << std::endl;}

			for( ; acceptedCandidates_.size() < nSamples && totalPoints < maxPoints; totalPoints++ ){

					yCurr = func_( x_ );
				//Mod 14Nov2019
				//	if( yCurr < 0 ){

				//	    std::cout << "\n\nError: x_ current is in negative domain" << "\n\n";
				//	    exit(0);
				//	}

					xProp = transitionKernel_();

					yProp = func_( xProp );

					for( int i = 0 ; i < DIM ; i++ ){

						if( xProp[i] < argBounds_.first[i] || xProp[i] > argBounds_.second[i] ){

							yProp = 0;
							break;
						}
					}
					//Mod 14 Nov 2019
					//if( yProp < 0 ){

                   //    std::cout << "Proposed values is nagative, set boudaries" << '\n';
                   //     exit(0);
                   // }
					//Mod 14 Nov 2019
					//Ratio   = std::min( 0.0 , log( yProp ) - log( yCurr ) );
					Ratio   = std::min( 0.0 ,  yProp  -  yCurr  );
					randNum = log( uniformDistribution( engine ) + (double) 1e-6);

					if( Ratio > randNum ){

						x_ = xProp;

						if ( yProp > valMax_ ){
								valMax_ = yProp;
								xMax_   = x_;
							}

						if( totalPoints > burnInNum_ ){

							acceptedCandidates_.push_back( std::make_pair( x_, yProp ) );
						}

					}else{
					    rejectedNum_++;
					}
			}
			break;


	case OTHER:
		std::cout << "This Choice has not been added yet!" << std::endl;
		exit(0);
	}


	//try and catch in case of invalid division
	try{

		acceptanceRatio_ = (double) acceptedCandidates_.size() / totalPoints;
	}
	catch( double acRError ){

		std::cout << "Error Calculating acceptance Ratio" << std::endl;
	}

	if(classVerbosity_)
	{
	    std::cout << "acceptanceRatio_ = :" << acceptanceRatio_ << '\n';
	    std::cout << "acceptedCandidates_.size() in sample = "
			      << acceptedCandidates_.size() << std::endl;
	}
	return;
}


template<unsigned DIM, typename FUNC, typename ARG>
void MCMC<DIM,FUNC,ARG>::setSigma( ARG sigmaJump, const unsigned maxStep, unsigned optSampleSize,
								   const bool setSigmaVerbose){

	/*Method to setSigma if sigma optimal is not know
	 *This will iteratively call sample() and ajust the sigma vector in proportions to
	 *its initialisation values. i.e. if you want larger sigma in one direction, then set initial sigma
	 *larger in that direction from constructor.*/

	//if verbose true save the evolution of sigma with respect to the acceptance ratio

	if( setSigmaVerbose ){ allSigmaJumps_.reserve ( 0.25 * maxStep ) ;}

	ARG sigmaBase = sigmaStep_;




	// resource for this numbers of optimal acceptance
	//https://m-clark.github.io/docs/ld_mcmc/index_onepage.html#am
	/*BÉDARD, M. (2008). Optimal acceptance rates for Metropolis algorithms:
	  Moving beyond 0.234. Stochastic Process. Appl. 118 2198–2222. MR2474348*/
	double accOpt = 0.0;
	double accTol = 0.0;

	if( DIM < 5 ){

		accOpt = 0.44;
		accTol =  0.1;
	}
	else if( DIM >= 5 ){

		accOpt = 0.23;
		accTol = 0.05;
	}
	else{

		std::cerr << "Not implemented for the given DIM\n";
		exit(1);
	}

	bool Optimized = false;
	unsigned ctr = 0;

	while ( (!Optimized) && ( ctr < maxStep  ) ){

		if( classVerbosity_ ){

			std::cout << "\n\n";
			std::cout << "Optimising sigma, loop: " << ctr << '\n';
			std::cout << "sigmaStep_ = : \n"<< std::endl << "\t";

			for( int i =0; i < DIM; ++i){

			    std::cout << sigmaStep_[i] << " ";
			}

			std::cout << '\n';
		}

		//User defined optimisation sample size
		sample( optSampleSize );

		//Test if above or under acceptance limits, if not optimised yet, apply one sided bisection method.
		//New sigmaSteps_ proportional to initialisation sizes of the sigma vector

		if( acceptanceRatio_ > ( accTol+ accOpt ) ){

			sigmaBase = sigmaStep_;

			//modifi values of sigma one by one
			for ( int i=0; i < DIM; ++i ){
			    //sigma too small, augment next sigmaStep_

				sigmaStep_[i] = sigmaBase[i] + sigmaJump[i];
			}
		}
		else if(acceptanceRatio_ < ( accOpt - accTol ) ){

			for( unsigned int i=0; i <DIM; ++i ){
			    //sigma went to far, split jump in half an try new sigmaStep_

				sigmaJump[i]  = sigmaJump[i] / 2.0;
				sigmaStep_[i] = sigmaBase[i] + sigmaJump[i];
			}
		}
		else if( acceptanceRatio_< (accOpt + accTol ) && ( acceptanceRatio_ > accOpt-accTol ) ){
		    //condition for having optimized sigma

			Optimized = true;

			if( classVerbosity_ ){

				std::cout<<"sigma optimized to: ";

				for(int i =0; i < DIM;i++){

					std::cout<<sigmaStep_[i]<<" ";
				}
			}
				std::cout<<std::endl;
		}
		else{
		    std::cout << "Error is Optimizing sigma" << std::endl;
		}
		ctr++;

		if( classVerbosity_ ){

					if( ctr >= maxStep ){
						//std::vector< std::pair < ARG, double > > allSigmaJumps;

						std::cout<<"did not reach optimal sig!!\n";

						std::cout<<"Number of Loops Performed : " << allSigmaJumps_.size() - 1 << '\n';
					}
			}

		if( setSigmaVerbose ){

			allSigmaJumps_.push_back( std::make_pair( sigmaStep_, acceptanceRatio_ ) );
		}

		}

	return;
}


template<unsigned DIM, typename FUNC, typename ARG>
void MCMC<DIM,FUNC,ARG>::writeResult( std::ostream & out ){

	//Simple mehtod to save results of accepted candidates.

	//Ref of data type:
	//std::vector< std::pair<ARG, double> >	acceptedCandidates_;

	for( unsigned i = 0; i < acceptedCandidates_.size(); ++i ){

		for(int j = 0; j < DIM; ++j ) {

			out << acceptedCandidates_[i].first[j] << " ";
		}
		out << acceptedCandidates_[i].second << '\n';
	}
}


#endif // MCMC_HPP_


