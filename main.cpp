#include"algorithm.hpp"

int main(){
    RidgeRegresion R("abalone.txt");
    RidgeRegresion *Rp = &R;
    GradientDescent GD(2000,0);
    cout<<R.MSE()<<endl;
    GD.optimize(Rp);
    cout<<R.MSE()<<endl;
}