#include<iostream>
#include<vector>
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
    double lambda;//hyper parameter
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
class GradientDescent:public optimizer{
    GradientDescent(int loops, double stop):optimizer(loops,stop){

    }
};
class ConjugateDescent:optimizer{
    ConjugateDescent(int loops, double stop):optimizer(loops,stop){

    }
};
class quasiNewton:optimizer{
    quasiNewton(int loops, double stop):optimizer(loops,stop){

    }
};
