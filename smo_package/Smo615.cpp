#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cstdlib>
#include<climits>
#include<numeric>
#include<cmath>
#include<boost/random/mersenne_twister.hpp>
#include<boost/random/uniform_int.hpp>
#include<Eigen/Dense>
#include<boost/tokenizer.hpp>
#include<fstream>
#include<iomanip>
#include<boost/lexical_cast.hpp>
#include "ReadMatrix615.h"
using namespace std;
using namespace Eigen;
using namespace boost;
#define tol 0.01

MatrixXd standardize(MatrixXd X);
double linear_kernel(VectorXd x, VectorXd y, double b);
double gaussian_kernel(VectorXd x, VectorXd y, double sigma);
MatrixXd linear_kernel_matrix(VectorXd x, VectorXd y, double b);
MatrixXd gaussian_kernel_matrix(VectorXd x, VectorXd y, double sigma);
double objective_function(VectorXd alpha, VectorXd target, int ind, MatrixXd X_train);
VectorXd decision_function(VectorXd alphas, VectorXd target, int ind, MatrixXd X_train, MatrixXd x_test, double b);
double decision_function_double(VectorXd alphas, VectorXd target, int ind, MatrixXd X_train, MatrixXd x_test, double b);
vector<double> linspace(double a, double b, int N);

class SMOModel {
public:
    MatrixXd X;
    VectorXd y;
    double C;    
    VectorXd alphas;
    double b;
    VectorXd errors;
    int m;
    int ind;
     
    SMOModel(MatrixXd M, VectorXd l, double c, VectorXd a, double bias, VectorXd err, int indicator);
    void train();
    int examine_example(int i2);
    int take_step(int i1, int i2);
    VectorXd predict(MatrixXd test);

};

double linear_kernel(VectorXd x, VectorXd y, double b) {
    return (x.transpose() * y + b);
}

double gaussian_kernel(VectorXd x, VectorXd y, double sigma) {
    double sum = 0;
    for (int i = 0; i < (int)x.size(); ++i) {
        sum += pow((x[i] - y[i]), 2);
    }
    //cout <<(exp(-1.0 * sum / (2 * pow(sigma, 2))))<< endl;
    return (exp(-1.0 * sum / (2 * pow(sigma, 2))));             
}

MatrixXd linear_kernel_matrix(MatrixXd x, MatrixXd y, double b) {
    MatrixXd one = b * MatrixXd::Ones(x.rows(), y.rows());
    return (x * y.transpose() + one);
}

MatrixXd gaussian_kernel_matrix(MatrixXd x, MatrixXd xy, double sigma) {
    MatrixXd gaussian((int)x.rows(), (int)xy.rows());
    MatrixXd y;
    if ((int)xy.cols() == 1) {
        gaussian.resize((int)x.rows(),1);
        y = xy.transpose(); 
    }
    else {
        y = xy;
    }
    
    VectorXd row_vec;
    //cout << x.rows() << " " << y.rows() << endl;
    for (int i = 0; i < (int) x.rows(); i++) {
        for (int j = 0; j < (int) y.rows(); j++) {
            row_vec = x.row(i) - y.row(j);
            double tmp = 0;
            for (int k = 0; k < (int) row_vec.size(); k++) {
                tmp += pow(row_vec[k], 2);
            }
            gaussian(i, j) = exp(-1.0 * tmp / (2 * pow(sigma, 2)));
        }
    }  
    return gaussian;
}

double objective_function(VectorXd alphas, VectorXd target, int ind, MatrixXd X_train) {
    double sum = 0;
    MatrixXd kernel;
    if (ind < 0.5) {
        kernel = linear_kernel_matrix(X_train, X_train, 1.0);
        
    } else {
        kernel = gaussian_kernel_matrix(X_train, X_train, 1.0);
    }
    for (int i = 0; i < (int)(alphas.size()); i++) {
        for (int j = 0; j < (int)(alphas.size()); j++) {
            sum += target[i] * alphas[i] * target[j] * alphas[j] * kernel.row(i) * kernel.row(j).transpose();
        }
    }

    return (alphas.sum() - 0.5 * sum);
}
    
VectorXd decision_function(VectorXd alphas, VectorXd target, int ind, MatrixXd X_train, MatrixXd x_test, double b) {
    MatrixXd kernel;
    if (ind < 0.5) {
        kernel = linear_kernel_matrix(X_train, x_test, 1.0);
    } else {
        kernel = gaussian_kernel_matrix(X_train, x_test, 1.0);
        //cout << kernel * (1.0 * alphas.cwiseProduct(target)) << endl;
    }
    return (kernel * (1.0 * alphas.cwiseProduct(target)) - 1.0 * b * VectorXd::Ones((int)(alphas.size())));
}

double decision_function_double(VectorXd alphas, VectorXd target, int ind, MatrixXd X_train, MatrixXd x_test, double b) {
    MatrixXd kernel;
    if (ind < 0.5) {
        kernel = linear_kernel_matrix(X_train, x_test.transpose(), 1.0);
    } else {
        kernel = gaussian_kernel_matrix(X_train, x_test, 1.0);
        //cout << kernel * (1.0 * alphas.cwiseProduct(target)) << endl;
    }
    double res = (kernel.transpose() * (1.0 * alphas.cwiseProduct(target)))[0];

    return (res - b);
}

SMOModel::SMOModel(MatrixXd M, VectorXd l, double c, VectorXd a, double bias, VectorXd err, int indicator) {
    X = M;
    y = l;
    C = c;
    alphas = a;
    b = bias;
    errors = err;
    m = (int)y.size();
    ind = indicator;
}

void SMOModel::train() {
    int numChanged = 0;
    int examineAll = 1;
    while ((numChanged > 0) || (examineAll != 0)) {
        numChanged = 0;
        if (examineAll != 0) {
            for (int i = 0; i < (int)alphas.size(); ++i) {
                int examine_result = examine_example(i);
                numChanged += examine_result;
            }
        } else {
            vector<int> index;
            for (int i = 0; i < (int)alphas.size(); ++i) {
                if (alphas[i] != 0 && alphas[i] != C) {
                    index.push_back(i);
                }
            }
            for (int i = 0; i < (int)index.size(); ++i) {
                int examine_result = examine_example(index[i]);
                numChanged += examine_result;
            }    
        }
        if (examineAll == 1) {
            examineAll = 0;
        } else if (numChanged == 0) {
            examineAll = 1;
        }
    }
}

int SMOModel::examine_example(int i2) {
    int y2 = y[i2];
    double alph2 = alphas[i2];
    double E2 = errors[i2];
    double r2 = E2 * y2;
    int i1 = 0;
    int step_result;
    vector<int> where;
    int len = 0;
    int rows = (int)alphas.size();
    
    // Proceed if error is within specified tolerance (tol)
    if (((r2 < (-tol)) && (alph2 < C))||((r2 > tol) && (alph2 > 0))) {
        for (int i = 0 ; i < rows; ++i) {
            if ((alphas[i] != 0) && (alphas[i] != C)) {
                where.push_back(i);
                len++;
            }
        }
        if (len > 1) {
            //Use 2nd choice heuristic is choose max difference in error
            if (errors[i2]>0) {
                for(int i = 0; i < (int)errors.size(); ++i){
                    if(errors[i] == errors.minCoeff()){
                       i1 = i;
                       break;
                     }
                 }
            }
            else{
                 for(int i = 0; i < (int)errors.size(); ++i){
                    if(errors[i] == errors.maxCoeff()){
                       i1 = i;
                       break;
                     }
                 }
            }
            //cout << i1 << endl; 
            step_result = take_step(i1, i2);
            if (step_result) {
                return 1;
            }
        }   
        // Loop through non-zero and non-C alphas, starting at a random point         
        mt19937 rng;
        uniform_int<> ran(0, m-1);
        int x = ran(rng);
        int num = (int)where.size();
        if (num != 0) {
            int r = x % num;
            vector<int> roll1;

            for (int i = num - r; i < num; ++i) {
                roll1.push_back(where[i]);
            }
            for (int i = 0; i < num - r; ++i) {
                roll1.push_back(where[i]);
            }
            for (int i = 0; i < num; ++i) {
                i1 = roll1[i];
                step_result = take_step(i1, i2);
                if (step_result) {
                    return 1;
                }
            }
        }
        // loop through all alphas, starting at a random point
        int rand = ran(rng);
        vector<int> roll2;
        for (int i = m-rand; i < m; ++i) {
            roll2.push_back(i);
        }
        for (int i = 0; i < m-rand; ++i) {
            roll2.push_back(i);
        }
        for (int i = 0; i < m; ++i) {
            i1 = roll2[i];
            step_result = take_step(i1, i2);
            if (step_result) {
                return 1;
            }
        }
    }
    return 0;
}


int SMOModel::take_step(int i1, int i2) {
    //cout << i1 << " " << i2 << endl;
    if(i1==i2) {
        return 0;
    }
    double eps = 0.01;
    double alph1 = alphas[i1];
    double alph2 = alphas[i2];
    int y1 = y[i1];
    int y2 = y[i2];
    double E1 = errors[i1];
    double E2 = errors[i2];
    int s = y1 * y2;
    double L, H, eta;
    double k11, k12, k22, a2;

    if(y1 != y2) {
        L = max(0.0, alph2 - alph1);
        H = min(C, C + alph2 - alph1);
    } else {
        L = max(0.0, alph2 + alph1 - C);
        H = min(C, alph2 + alph1);
    }

    if (L == H) {
        return 0;
    }

    if (ind < 0.5) {
        k11 = linear_kernel(X.row(i1), X.row(i1), 1.0);
        k12 = linear_kernel(X.row(i1), X.row(i2), 1.0);
        k22 = linear_kernel(X.row(i2), X.row(i2), 1.0);
        eta = 2 * k12 - k11 - k22;
    } else {
        k11 = gaussian_kernel(X.row(i1), X.row(i1), 1.0);
        //cout << k11 << endl;
        k12 = gaussian_kernel(X.row(i1), X.row(i2), 1.0);
        k22 = gaussian_kernel(X.row(i2), X.row(i2), 1.0);
        eta = 2 * k12 - k11 - k22;      
    }
    
    if (eta < 0) {
        a2 = alph2 - y2 * (E1 - E2) / eta;
        if (a2 <= L) {
            a2 = L;
        } else if (a2 >= H) {
            a2 = H;
        //} else if (L < a2 && a2 < H) {
        //    continue;
        }
    } else {
        VectorXd alphas_adj = alphas;
        alphas_adj[i2] = L;
        double Lobj = objective_function(alphas_adj, y, ind, X);
        alphas_adj[i2] = H;
        double Hobj = objective_function(alphas_adj, y, ind, X);

        if(Lobj > (Hobj + eps)) {
            a2 = L;
        } else if (Lobj < (Hobj - eps)) {
            a2 = H;
        } else {
            a2 = alph2;
        }
    }

    if(a2 < 1e-08) {
        a2 = 0.0;
    } else if(a2 > (C - 1e-08)) {
        a2 = C;
    }

    if(fabs(a2 - alph2) < eps * (a2 + alph2 + eps)) {
        return 0;
    }

    double a1 = alph1 + s * (alph2 - a2);
    double b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + b;
    double b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + b;
    double b_new;

    if ((0 < a1) && (a1 < C)) {
        b_new = b1;
    } else if (0 < a2 && a2 < C) {
        b_new = b2;
    } else {
        b_new = (b1 + b2) * 0.5;
    }

    alphas[i1] = a1;
    alphas[i2] = a2;

    if ((0 < a1) && (a1 < C)) errors[i1] = 0;
    if ((0 < a2) && (a2 < C)) errors[i2] = 0;

    for(int i = 0; i < m; i++) {
        if (ind < 0.5) {
            if ((i != i1) && (i != i2)) {
                errors[i] = errors[i] + y1*(a1-alph1)*linear_kernel(X.row(i1), X.row(i), 1.0) + y2*(a2-alph2)*linear_kernel(X.row(i2),X.row(i), 1.0) + b - b_new;
            }
        } else {
            if ((i != i1) && (i != i2)) {
                errors[i] = errors[i] + y1*(a1-alph1)*gaussian_kernel(X.row(i1), X.row(i), 1.0) + y2*(a2-alph2)*gaussian_kernel(X.row(i2),X.row(i), 1.0) + b - b_new;
            }
        }
    }
    b = b_new;

    return 1;
}

VectorXd SMOModel::predict(MatrixXd test) {
    VectorXd w = VectorXd::Zero(X.cols()); 
    VectorXd one = VectorXd::Ones(test.rows());
    MatrixXd tmp;
    VectorXd res;
    if (ind == 0) {
        for (int i = 0; i < (int)y.size(); i++) {
            w += alphas[i] * y[i] * X.row(i); 
        }
        //cout << alphas << endl;
        res = test * w - b * one;
    }
    else {
        w.resize(test.rows());
        for (int j = 0; j < (int)y.size(); j++) {
            tmp = gaussian_kernel_matrix(test, X.row(j).transpose(), 1.0);
            w += alphas[j] * y[j] * tmp.col(0); 
        }
        //cout << alphas << endl;
        res = w - b * one;
    }
    return res;  
}

MatrixXd standardize(MatrixXd X) {
    double sum;
    int size = (int)X.rows();
    vector<double> mu, sig;
    for (int i = 0; i < (int)X.cols(); i++) {
        sum = 0.0;
        mu.push_back(1.0 * X.col(i).sum() / size);
        for (int j = 0; j < size; j++) {
            sum += pow(X.col(i)[j], 2);
        }
        sig.push_back(pow((1.0 / size) * (sum - size * mu[i] * mu[i]), 0.5));
        for (int k = 0; k < size; k++) {
            X.col(i)[k] = (1.0 * X.col(i)[k] - mu[i]) / sig[i];
        }
    }
    return X; 
}

vector<double> linspace(double a, double b, int N) {
    double h = (b - a) / static_cast<double>(N-1);
    vector<double> xs(N);
    vector<double>::iterator x;
    double val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;
    return xs;
}

extern "C" {
    void smo615(char const **train_data, char const **train_label, char const **test_data, char const **test_label, double* c, int ind, double* grid, double* wb) {
        MatrixXd init_X = readFromFile<double>(*train_data);
        MatrixXd init_y = readFromFile<double>(*train_label);
        MatrixXd X_test = readFromFile<double>(*test_data);
        VectorXd y_test = readFromFile<double>(*test_label);
        VectorXd labels = init_y.col(0);
        VectorXd init_alphas = VectorXd::Zero((int)labels.size());

        for (int i = 0; i < (int)labels.size(); i++) {
            if (labels[i] == 0)
                labels[i] = -1;
        }

        for (int i = 0; i < (int)y_test.size(); i++) {
            if (y_test[i] == 0)
                y_test[i] = -1;
        }
        
        init_X = standardize(init_X);
        X_test = standardize(X_test);
        
        SMOModel model(init_X, labels, *c, init_alphas, 0.0, init_alphas, ind);
        
        VectorXd init_error = decision_function(model.alphas, model.y, ind, model.X, model.X, model.b) - model.y;

        model.errors = init_error;
     
        model.train();
        
        VectorXd y_pred = model.predict(X_test);
        
        double accuracy = 0;
        for (int i = 0; i < (int)X_test.rows(); i++) {
            if (((double)y_pred[i] * (double)y_test[i]) > 0) {
                accuracy += 1.0;
            }
        }

        cout << setprecision(3) << "Accuracy: " << accuracy / (int)X_test.rows() << endl; 

        int count = 0;
        vector<double> xrange = linspace(model.X.col(0).minCoeff(),model.X.col(0).maxCoeff(),100);
        vector<double> yrange = linspace(model.X.col(1).minCoeff(),model.X.col(1).maxCoeff(),100);
        MatrixXd xy(2,1);
        
        // Plot
        for(int i = 0; i < xrange.size(); i++) {
            for(int j = 0; j < yrange.size(); j++) {
                xy(0,0) = xrange[i];
                xy(1,0) = yrange[j];
                //cout << xy << endl;
                //grid.row(i) = decision_function(model.alphas, model.y, 1, model.X, xy, model.b);
                grid[count] = decision_function_double(model.alphas, model.y, ind, model.X, xy, model.b);
                count++;
                //grid.push_back(res);
            }
        }
        VectorXd w = VectorXd::Zero(model.X.cols());
        for (int j = 0; j < (int)model.y.size(); j++) {
            w += model.alphas[j] * model.y[j] * model.X.row(j); 
        }
        wb[0] = w[0];
        wb[1] = w[1];
        wb[2] = model.b;
        
    }  
}
