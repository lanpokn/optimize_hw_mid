#include"algorithm.hpp"

int main(){
    RidgeRegresion R1("abalone.txt");
    RidgeRegresion R2("abalone.txt");
    RidgeRegresion R3("abalone.txt");
    RidgeRegresion *Rp = &R1;
    GradientDescent GD(20,0);
    GD.optimize(Rp);
    Rp = &R2;
    ConjugateDescent CD(20,0);
    CD.optimize(Rp);
    quasiNewton QN(20,0);
    Rp = &R3;
    QN.optimize(Rp);
    cout<<R1.MSE()<<endl;
}