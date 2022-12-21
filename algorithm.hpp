#include<iostream>
#include<vector>
#include<fstream>
#include <sstream> 
#include<boost/algorithm/string.hpp>
#include<eigen3/Eigen/Core>
using namespace std;
string DoubleToString(double Input) 
{ 
    stringstream Oss; 
    Oss<<Input; 
    return Oss.str(); 
} 
 
double StringToDouble(string Input) 
{ 
    double Result; 
    stringstream Oss; 
    Oss<<Input; 
    Oss>>Result; 
    return Result; 
}

/**
 * @brief 岭回归用优化方法求解，本质上是一次性的读取进来一堆参数，整理成X和Y，用这些参数
 *        生成一个只和系数theta有关的损失函数J(theta),然后用各种方法处理J(theta),
 *        找到能让J最小的theta值，注意此时不涉及任何读入新数据的工作。神经网络那种读一点数据，反向传播一次那种思路并不适合简单的岭回归
 *        所以基本思路：该类读取数据，定义输入和输出的基本信息
 *        内部实现：beta作为线性拟合系数，有常数项，从0到n，是输入维度加1，lambda不好调整，直接定位常数，即对所有的都相同
 *        提供接口：  系数信息（主要是维度，系数要在optimizer里存储）
 *                  损失函数，输入一组系数，输出损失函数值
 */
class RidgeRegresion{
public:
    double lambda = 0.5;//hyper parameter
    std::vector<double> B = {1,1,1,1,1,1,1,1};//
    int size = 0;
    double Y[10000];
    double X[8][10000]; 
    RidgeRegresion(std::string filepath){
        std::fstream file(filepath);
        std::string line;
        std::vector<std::string> words;
        int i = 0;
        while(std::getline(file, line)){
            boost::split(words, line, boost::is_any_of(" "), boost::token_compress_on);
            int j = 0;
            Y[i] = StringToDouble(words[0]);
            j++;
            if(words[1][0] == '2'){
                X[0][i] = 0;//没有数据一律补0处理
                for( ;j<8;j++){
                    words[j] = words[j].substr(2);
                    X[j][i] = StringToDouble(words[j]);
                }
            }
            else{
                for( ;j<9;j++){
                    words[j] = words[j].substr(2);
                    X[j-1][i] = StringToDouble(words[j]);
                }
            }
            size++;
            i++;
        }
    }
    /**
     * @brief print the MSE of the result+
     * 
     * @return double 
     */
    double MSE(){
        double ret = 0;
        for(int i =0;i<size;i++){
            double error =Y[i];
            for(int j = 0; j<8;j++){
                error+= -(B[j]*X[j][i]);
            }
            ret+= error*error;
        }
        return ret;
    }
    // double Loss(){

    // }
};
/**
 * @brief optimizer的接口应该很易懂，要求维护一个虚函数，
 *        该函数以一个RidgeRegresion的指针作为输入
 *        运行之后RidegeREgresion的参数变为最优
 *        所以所有要增加的都在该类增加，子类只用来维护那个函数的具体实现
 */
class optimizer{
protected:
    int largest_loops = 100;
    double stop_error = 0.1;
public:
    optimizer(int loops, double stop){
        largest_loops = loops;
        stop_error = stop;
    }
    virtual void optimize(RidgeRegresion* system) = 0;
};
/**
 * @brief 注意delta_L就是该店的梯度，在各种方法中应该通用，所以就不改了。
 * 
 */
class GradientDescent:public optimizer{
public:
    double rate = 0.09;
    GradientDescent(int loops, double stop):optimizer(loops,stop){

    }
    virtual void optimize(RidgeRegresion* system){
        for(int i = 0; i<largest_loops;i++){
            //below are the conclusion of the method
            double delta_L[8];
            for(int j=0;j<8;j++){
                delta_L[j] = 0;
                for(int k =0;k<system->size;k++){
                    delta_L[j]+=-2*(system->Y[k]-system->B[0]*system->X[0][k]);
                    delta_L[j]+=-2*(-system->B[1]*system->X[1][k]);
                    delta_L[j]+=-2*(-system->B[2]*system->X[2][k]);
                    delta_L[j]+=-2*(-system->B[3]*system->X[3][k]);
                    delta_L[j]+=-2*(-system->B[4]*system->X[4][k]);
                    delta_L[j]+=-2*(-system->B[5]*system->X[5][k]);
                    delta_L[j]+=-2*(-system->B[6]*system->X[6][k]);
                    delta_L[j]+=-2*(-system->B[7]*system->X[7][k]);
                    delta_L[j]*=system->X[j][k];
                }
                delta_L[j]+=2*system->lambda*system->B[j];
            }
            for(int j = 0;j<8;j++){
                system->B[j]-=rate*delta_L[j];
            }
            // rate =  rate;
        }
    }
};
class ConjugateDescent:optimizer{
    ConjugateDescent(int loops, double stop):optimizer(loops,stop){

    }
    double rate =0.09;
    virtual void optimize(RidgeRegresion* system){
        for(int i = 0; i<largest_loops;i++){
        //below are the conclusion of the method
        //1
        double delta_L[8];
        double delta_L_old[8];
        double p[8];
        double x_old[8];
        for(int j=0;j<8;j++){
            delta_L[j] = 0;
            for(int k =0;k<system->size;k++){
                delta_L_old[j]+=-2*(system->Y[k]-system->B[0]*system->X[0][k]);
                delta_L_old[j]+=-2*(-system->B[1]*system->X[1][k]);
                delta_L_old[j]+=-2*(-system->B[2]*system->X[2][k]);
                delta_L_old[j]+=-2*(-system->B[3]*system->X[3][k]);
                delta_L_old[j]+=-2*(-system->B[4]*system->X[4][k]);
                delta_L_old[j]+=-2*(-system->B[5]*system->X[5][k]);
                delta_L_old[j]+=-2*(-system->B[6]*system->X[6][k]);
                delta_L_old[j]+=-2*(-system->B[7]*system->X[7][k]);
                delta_L_old[j]*=system->X[j][k];
            }
            delta_L_old[j]+=2*system->lambda*system->B[j];
            if(i == 0){
                p[j] = delta_L_old[j];
            }
        }
        for(int j=0;j<8;j++){
            x_old[j] = system->B[j];
            system->B[j]-=rate*p[j];
        }
        //2
        for(int j=0;j<8;j++){
            delta_L[j] = 0;
            for(int k =0;k<system->size;k++){
                delta_L[j]+=-2*(system->Y[k]-system->B[0]*system->X[0][k]);
                delta_L[j]+=-2*(-system->B[1]*system->X[1][k]);
                delta_L[j]+=-2*(-system->B[2]*system->X[2][k]);
                delta_L[j]+=-2*(-system->B[3]*system->X[3][k]);
                delta_L[j]+=-2*(-system->B[4]*system->X[4][k]);
                delta_L[j]+=-2*(-system->B[5]*system->X[5][k]);
                delta_L[j]+=-2*(-system->B[6]*system->X[6][k]);
                delta_L[j]+=-2*(-system->B[7]*system->X[7][k]);
                delta_L[j]*=system->X[j][k];
            }
            delta_L[j]+=2*system->lambda*system->B[j];
        }
    }
};
class quasiNewton:optimizer{
    quasiNewton(int loops, double stop):optimizer(loops,stop){

    }
};
