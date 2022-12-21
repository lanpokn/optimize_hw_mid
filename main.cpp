#include"algorithm.hpp"

int main(){
    RidgeRegresion R("abalone.txt");
    RidgeRegresion *Rp = &R;
    GradientDescent GD(20,0);
    cout<<R.MSE()<<endl;
    GD.optimize(Rp);
    cout<<R.MSE()<<endl;
}