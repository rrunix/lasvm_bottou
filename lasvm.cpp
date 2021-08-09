// -*- Mode: c++; c-file-style: "stroustrup"; -*-

#include <stdio.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <Rcpp.h>
using namespace Rcpp;

using namespace std;


#include "vector.h"
#include "lasvm.h"

#define LINEAR  0
#define POLY    1
#define RBF     2
#define SIGMOID 3 

#define ONLINE 0
#define ONLINE_WITH_FINISHING 1

#define RANDOM 0
#define GRADIENT 1
#define MARGIN 2

#define ITERATIONS 0
#define SVS 1
#define TIME 2

const char *kernel_type_table[] = {"linear","polynomial","rbf","sigmoid"};

class stopwatch
{
public:
    stopwatch() : start(std::clock()){} //start counting time
    ~stopwatch();
    double get_time() 
        {
            clock_t total = clock()-start;;
            return double(total)/CLOCKS_PER_SEC;
        };
private:
    std::clock_t start;
};
stopwatch::~stopwatch()
{
//    clock_t total = clock()-start; //get elapsed time
//     if (verbosity > 0)
// 		Rcout<<"Time(secs): "<<double(total)/CLOCKS_PER_SEC<<endl;
}
class ID // class to hold split file indices and labels
{
public:
    int x;
    int y;
    ID() : x(0), y(0) {}
    ID(int x1,int y1) : x(x1), y(y1) {}
};
// IDs will be sorted by index, not by label.
bool operator<(const ID& x, const ID& y)
{
    return x.x < y.x;
}



/* Data and model */
int m=0;                          // training set size
vector <lasvm_sparsevector_t*> X; // feature vectors
vector <int> Y;                   // labels
vector <double> kparam;           // kernel parameters
vector <double> alpha;            // alpha_i, SV weights
double b0;                        // threshold

/* Hyperparameters */
int kernel_type=RBF;              // LINEAR, POLY, RBF or SIGMOID kernels
double degree=3,kgamma=-1,coef0=0;// kernel params
int use_b0=1;                     // use threshold via constraint \sum a_i y_i =0
int selection_type=RANDOM;        // RANDOM, GRADIENT or MARGIN selection strategies
int optimizer=ONLINE_WITH_FINISHING; // strategy of optimization
double C=1;                       // C, penalty on errors
double C_neg=1;                   // C-Weighting for negative examples
double C_pos=1;                   // C-Weighting for positive examples
int epochs=1;                     // epochs of online learning
int candidates=50;				  // number of candidates for "active" selection process
double deltamax=1000;			  // tolerance for performing reprocess step, 1000=1 reprocess only
vector <double> select_size;      // Max number of SVs to take with selection strategy (for early stopping) 
vector <double> x_square;         // norms of input vectors, used for RBF

/* Programm behaviour*/
int verbosity=0;                  // verbosity level, 0=off
int saves=1;
char report_file_name[1024];             // filename for the training report
char split_file_name[1024]="\0";         // filename for the splits
int cache_size=256;                       // 256Mb cache size as default
double epsgr=1e-3;                       // tolerance on gradients
size_t kcalcs=0;                      // number of kernel evaluations
int binary_files=0;
vector <ID> splits;             
int max_index=0;
vector <int> iold, inew;		  // sets of old (already seen) points + new (unseen) points
int termination_type=0;


void exit_with_help()
{
	stop ("Command line had some error.");
}


void la_svm_parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
    int i; int clss; double weight;
	select_size.clear();
	
    // parse options
    for(i=1;i<argc;i++) 
    {
//		Rcout << argv[i][0] << " ";
		if(argv[i][0] != '-') break;
        ++i;
//		Rcout  << argv[i-1][1] << " " << argv[i] << "       ";
		switch(argv[i-1][1])
        {
			case 'o':
            optimizer = atoi(argv[i]);
            break;
        case 't':
            kernel_type = atoi(argv[i]);
            break;
        case 's':
            selection_type = atoi(argv[i]);
            break;
        case 'l':
            while(1)
            {
                select_size.push_back(atof(argv[i]));
                ++i; if((argv[i][0]<'0') || (argv[i][0]>'9')) break; 
            }
            i--;
            break;
        case 'd':
            degree = atof(argv[i]);
            break;
        case 'g':
            kgamma = atof(argv[i]);
            break;
        case 'r':
            coef0 = atof(argv[i]);
            break;
        case 'm':
            cache_size = (int) atof(argv[i]);
            break;
        case 'c':
            C = atof(argv[i]);
            break;
        case 'w':
            clss= atoi(&argv[i-1][2]);
            weight = atof(argv[i]);
            if (clss>=1) C_pos=weight; else C_neg=weight;
            break;
        case 'b':
            use_b0=atoi(argv[i]);
            break;
        case 'B':
            binary_files=atoi(argv[i]);
            break;
        case 'e':
            epsgr = atof(argv[i]);
            break;
        case 'p':
            epochs = atoi(argv[i]);
            break;
        case 'D':
            deltamax = atoi(argv[i]);
            break;
        case 'C':
            candidates = atoi(argv[i]);
            break;
        case 'T':
            termination_type = atoi(argv[i]);
            break;
        default:
            stop ("unknown command line option\n");
            exit_with_help();
        }
    }

    if (verbosity > 0)
		Rcout << "\tepochs: \t" << epochs << "\n";
    
    saves=select_size.size(); 
    if(saves==0) select_size.push_back(100000000);

	if (saves != 1)
		stop ("Sorry, lasvmR does not support more than one save for now.");
}


int split_file_load(char *f)
{
	stop ("should never be called directly.");
	return 0; // make everyone happy
}


int libsvm_load_data(char *filename)
// loads the same format as LIBSVM
{
	stop ("should never be called directly.");
	return 0; // make everyone happy
}


int binary_load_data(char *filename)
{
	stop ("should never be called directly.");
	return 0; // make everyone happy
}


void load_data_file(char *filename)
{
	stop ("should never be called directly.");
}


void adapt_data(int msz)
{
	int i;
	splits.resize(0); 
	
	if(kernel_type==RBF)
	{
		x_square.resize(m+msz);
		for(i=0;i<msz;i++)
			x_square[i+m]=lasvm_sparsevector_dot_product(X[i+m],X[i+m]);
	}
	
	if(kgamma==-1)
		kgamma=1.0/ ((double) max_index); // same default as LIBSVM
		
		m+=msz;
}



int sv1,sv2; double max_alpha,alpha_tol;

void resetVars() {
	
	/* Data and model */
	m=0;                          // training set size
	X.clear(); // feature vectors
	Y.clear();                   // labels
	kparam.clear();           // kernel parameters
	alpha.clear();            // alpha_i, SV weights
	b0 = 1;                        // threshold
	
	/* Hyperparameters */
	kernel_type=RBF;              // LINEAR, POLY, RBF or SIGMOID kernels
	degree=3;
	kgamma=-1;
	coef0=0;// kernel params
	use_b0=1;                     // use threshold via constraint \sum a_i y_i =0
	selection_type=RANDOM;        // RANDOM, GRADIENT or MARGIN selection strategies
	optimizer=ONLINE_WITH_FINISHING; // strategy of optimization
	C=1;                       // C, penalty on errors
	C_neg=1;                   // C-Weighting for negative examples
	C_pos=1;                   // C-Weighting for positive examples
	epochs=1;                     // epochs of online learning
	candidates=50;				  // number of candidates for "active" selection process
	deltamax=1000;			  // tolerance for performing reprocess step, 1000=1 reprocess only
	select_size.clear();      // Max number of SVs to take with selection strategy (for early stopping) 
	x_square.clear();         // norms of input vectors, used for RBF
	
	/* Programm behaviour*/
	verbosity=0;                  // verbosity level, 0=off
	saves=1;
	cache_size=256;                       // 256Mb cache size as default
	epsgr=1e-3;                       // tolerance on gradients
	kcalcs=0;                      // number of kernel evaluations
	binary_files=0;
	splits.clear();             
	max_index=0;
	iold.clear();
	inew.clear();		  // sets of old (already seen) points + new (unseen) points
	termination_type=0;
	sv1 = 0;
	sv2 = 0;
	max_alpha = 0;
	alpha_tol = 0;
}	



int count_svs()
{
    int i; 
    max_alpha=0; 
    sv1=0;sv2=0;
    
    for(i=0;i<m;i++) 	// Count svs..   
    {
        if(alpha[i]>max_alpha) max_alpha=alpha[i];
        if(-alpha[i]>max_alpha) max_alpha=-alpha[i];
    }
       
    alpha_tol=max_alpha/1000.0;
    
    for(i=0;i<m;i++) 
    {
        if(Y[i]>0) 
        {
            if(alpha[i] >= alpha_tol) sv1++; 
        }
        else    
        {
            if(-alpha[i] >= alpha_tol) sv2++; 
        }            
    }
    return sv1+sv2;
}


int libsvm_save_model(const char *model_file_name)
    // saves the model in the same format as LIBSVM
{
	stop ("should never be called directly.");
	return 0; // make everyone happy
}

double kernel(int i, int j, void *kparam)
{
    double dot;
    kcalcs++;
    dot=lasvm_sparsevector_dot_product(X[i],X[j]);
    
    // sparse, linear kernel
    switch(kernel_type)
    {
    case LINEAR:
        return dot;
    case POLY:
        return pow(kgamma*dot+coef0,degree);
    case RBF:
        return exp(-kgamma*(x_square[i]+x_square[j]-2*dot));    
    case SIGMOID:
        return tanh(kgamma*dot+coef0);    
    }
    return 0;
} 
  


void finish(lasvm_t *sv)
{
    int i,l; 

    if (optimizer==ONLINE_WITH_FINISHING)
    {
		if (verbosity > 0)
			Rcout << "..[finishing]";
     
        int iter=0;

        do { 
            iter += lasvm_finish(sv, epsgr); 
        } while (lasvm_get_delta(sv)>epsgr);

    }

    l=(int) lasvm_get_l(sv);
    int *svind,svs; svind= new int[l];
    svs=lasvm_get_sv(sv,svind); 
    alpha.resize(m);
    for(i=0;i<m;i++) alpha[i]=0;
    double *svalpha; svalpha=new double[l];
    lasvm_get_alpha(sv,svalpha); 
    for(i=0;i<svs;i++) alpha[svind[i]]=svalpha[i];
    b0=lasvm_get_b(sv);
	delete[] svind;
	delete[] svalpha;
}



void make_old(int val)
    // move index <val> from new set into old set
{
    int i,ind=-1;
    for(i=0;i<(int)inew.size();i++)
    {
        if(inew[i]==val) {ind=i; break;}
    }

    if (ind>=0)
    {
        inew[ind]=inew[inew.size()-1];
        inew.pop_back();
        iold.push_back(val);
    }
}


int select(lasvm_t *sv) // selection strategy
{
    int s=-1;
    int t,i,r,j;
    double tmp,best; int ind=-1;

// 	std::cout << "SEL--\n";
    switch(selection_type)
    {
    case RANDOM:   // pick a random candidate
        s=R::runif (0,inew.size());
        break;

    case GRADIENT: // pick best gradient from 50 candidates
// 		std::cout << "GR ";
        j=candidates; 
		if ((int)inew.size()<j) 
			j=inew.size();
		r=R::runif (0,inew.size());
// 		std::cout << r << "/" << inew.size() << "\n";
        s=r;
        best=1e60; //?
        for(i=0;i<j;i++)
        {
// 			std::cout << "t, s:" << s;
            r=inew[s];
// 			std::cout << "r=" << r;
			tmp=lasvm_predict(sv, r);  
            tmp*=Y[r];
//             printf("\n%d: example %d   grad=%g\n",i,r,tmp);
            if (tmp<best) {
// 				std::cout << "*";
				best=tmp;
				ind=s;
			}
        	s=R::runif (0,inew.size());
// 			std::cout << "ns:" << s;
		}  
        s=ind;
        break;

    case MARGIN:  // pick closest to margin from 50 candidates
        j=candidates; if((int)inew.size()<j) j=inew.size();
		r=R::runif (0,inew.size());
        s=r;
        best=1e60;
        for(i=0;i<j;i++)
        {
            r=inew[s];
            tmp=lasvm_predict(sv, r);  
            if (tmp<0) tmp=-tmp; 
            //printf("%d: example %d   grad=%g\n",i,r,tmp);
            if(tmp<best) {best=tmp;ind=s;}
			s=R::runif (0,inew.size());
		}  
        s=ind;
        break;
    }
// 	std::cout << "s " << s << "\n";
// 	std::cout << "size: " << inew.size() << "\n";
    t=inew[s]; 
    inew[s]=inew[inew.size()-1];
    inew.pop_back();
    iold.push_back(t);
	
    //printf("(%d %d)\n",iold.size(),inew.size());
// 	std::cout << "F\n";
    return t;
}


void train_online(char *model_file_name, char *input_file_name)
{
    int t1,t2=0,i,s,l,j;
    double timer=0;
    stopwatch *sw; // start measuring time after loading is finished (not true for fulltime)
    sw=new stopwatch;    // save timing information
    char t[1000];
    strcpy(t,model_file_name);
    strcat(t,".time");
    	
    lasvm_kcache_t *kcache = lasvm_kcache_create(kernel, NULL);
    lasvm_kcache_set_maximum_size (kcache, cache_size*1024*1024);
    lasvm_t *sv = lasvm_create (kcache, use_b0, C*C_pos, C*C_neg);
	if (verbosity > 0)
		Rcout << "set cache size to " << cache_size << "\n";

    // everything is new when we start
    for(i=0;i<m;i++) inew.push_back(i);
    
    // first add 5 examples of each class, just to balance the initial set
    int c1=0,c2=0;
    for(i=0;i<m;i++)
    {
        if(Y[i]==1 && c1<5) {lasvm_process(sv,i,(double) Y[i]); c1++; make_old(i);}
        if(Y[i]==-1 && c2<5){lasvm_process(sv,i,(double) Y[i]); c2++; make_old(i);}
        if(c1==5 && c2==5) break;
    }
    
    for(j=0;j<epochs;j++)
    {
		Rcpp::checkUserInterrupt();
		int steps = 10;
		if (m < 2*steps)
			steps = m;
		for(i=0;i<m;i++)
        {
			if(inew.size()==0) break; // nothing more to select
            s=select(sv);            // selection strategy, select new point
            
            t1=lasvm_process(sv,s,(double) Y[s]);
            
			if (deltamax<=1000) // potentially multiple calls to reprocess..
            {
                //printf("%g %g\n",lasvm_get_delta(sv),deltamax);
                t2=lasvm_reprocess(sv,epsgr);// at least one call to reprocess
				// actally this does not work with multiple times, and we do not intent to anyway.
				//std::cout << "i=" << i << "  m=" << m << "  m/10=" << m/10 << "\n";
				if (i % steps == 1) 
				{
					if (termination_type==TIME && sw->get_time()>=select_size[0])
					{
						// goto is completely underrated. please do more gotos.
						goto goto80;
					}
				}	
				while (lasvm_get_delta(sv)>deltamax && deltamax<1000)
                {
                    t2=lasvm_reprocess(sv,epsgr);
				}
            }
            
            if (verbosity==2) 
            {
                l=(int) lasvm_get_l(sv);
                Rcout << "l=" << l << " process=" << t1 << " reprocess=" << t2 << "\n";
            }
            else
                if(verbosity==1) {
                    if( (i%100)==0) { 
						Rcout << ".." << i << flush; 
						Rcpp::checkUserInterrupt();
					}
				}
            
            l=(int) lasvm_get_l(sv);
			if   ( (termination_type==ITERATIONS && i==select_size[0]) 
					|| (termination_type==SVS && l>=select_size[0])
					|| (termination_type==TIME && sw->get_time()>=select_size[0])
				)
			{  
				select_size.pop_back();
			}
            if(select_size.size()==0) break; // early stopping, all intermediate models saved
        }

        inew.resize(0);iold.resize(0); // start again for next epoch..
        for(i=0;i<m;i++) inew.push_back(i);
    }

goto80: 
    if(saves<2) 
    {
        finish(sv); // if haven't done any intermediate saves, do final save
        timer+=sw->get_time();
        //f << m << " " << count_svs() << " " << kcalcs << " " << timer << endl;
    }

    if(verbosity>0) {
		Rcout << "\n";
		l=count_svs(); 
		Rcout << "nSVs=" << l << "\n";
		Rcout << "||w||^2=" << lasvm_get_w2(sv) << "\n";
		Rcout << "kcalcs=" << kcalcs << endl;
	}
	
    //f.close();
    lasvm_destroy(sv);
    lasvm_kcache_destroy(kcache);
	delete sw;
}


int la_svm_main (int argc, char **argv)  
{
    char input_file_name[1024];
    char model_file_name[1024];
	la_svm_parse_command_line (argc, argv, input_file_name, model_file_name);

	load_data_file(input_file_name);

    train_online(model_file_name, input_file_name);
    
    libsvm_save_model(model_file_name);
	return (-1);
}

