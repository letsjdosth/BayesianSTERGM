// 프로그램 실행: <Ctrl+F5> 또는 [디버그] > [디버깅하지 않고 시작] 메뉴
// 프로그램 디버그: <F5> 키 또는 [디버그] > [디버깅 시작] 메뉴
// 시작을 위한 팁: 
//   1. [솔루션 탐색기] 창을 사용하여 파일을 추가/관리합니다.
//   2. [팀 탐색기] 창을 사용하여 소스 제어에 연결합니다.
//   3. [출력] 창을 사용하여 빌드 출력 및 기타 메시지를 확인합니다.
//   4. [오류 목록] 창을 사용하여 오류를 봅니다.
//   5. [프로젝트] > [새 항목 추가]로 이동하여 새 코드 파일을 만들거나, [프로젝트] > [기존 항목 추가]로 이동하여 기존 코드 파일을 프로젝트에 추가합니다.
//   6. 나중에 이 프로젝트를 다시 열려면 [파일] > [열기] > [프로젝트]로 이동하고 .sln 파일을 선택합니다.

#include <vector>
#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;

class Network {
    private:
        Mat<int> netStructure;
        bool isDirected;
        int n_Node;
        int n_Edge;

        void updateNetworkInfo() {
            n_Node = netStructure.n_rows;
            n_Edge = edgeNum();
        }
        
        int edgeNum() {
            if (isDirected){
                return accu(netStructure);
            }
            else {
                return accu(netStructure) / 2;
            }
        }

        Col<int> nodeDegree() {
            Col<int> nodeDegree = sum(netStructure, 1);
            return nodeDegree;
        }
        
        Col<int> nodeDegreeDist() {
            Col<int> degreeDistRes;
            degreeDistRes.zeros(n_Node + 1);
        
            Col<int> netNodeDegree = nodeDegree();
            Col<int>::iterator it = netNodeDegree.begin();
            Col<int>::iterator it_end = netNodeDegree.end();
            for (; it != it_end; ++it) {
                degreeDistRes(*it)++;
            }
            return degreeDistRes;
        }

        int n_triangle() {
            if (isDirected) {
                int count_Triangle = 0;

                for (int i = 0; i < n_Node; i++)
                {
                    for (int j = 0; j < n_Node; j++)
                    {
                        for (int k = 0; k < n_Node; k++)
                        {
                            // check the triplet if 
                            // it satisfies the condition 
                            if (netStructure(i,j) && netStructure(j,k) && netStructure(k,i))
                                count_Triangle++;
                        }
                    }
                }
            } else {
                return trace((netStructure * netStructure) * netStructure)/6;
            }
        }

    public:
        Network(Mat<int> inputNet, bool isDirectedInput) {
            netStructure = inputNet;
            isDirected = isDirectedInput;
            updateNetworkInfo();
        }

        Network() {
            //빈 생성자
        }

        int get_n_Node() {
            return n_Node;
        }

        int get_n_Edge() {
            return n_Edge;
        }

        Mat<int> get_netStructure() {
            return netStructure;
        }

        void printSummary() {
            cout << "===========================" << endl;
            cout << netStructure << endl;
            cout << "isDirected :" << isDirected << endl;
            cout << "n_node :" << n_Node << endl;
            cout << "n_edge :" << n_Edge << endl;
            cout << "node degree : " << nodeDegree().t() << endl;
            cout << "degree_dist : " << nodeDegreeDist().t() << endl;
            cout << "triangle : " << n_triangle() << endl;
            cout << "===========================" << endl;
        }

        
};

class netMCMCSampler {
    //undirected
private:
    Row<double> given_param;
    vector<Network> MCMCSampleVec;
    int n_iter;

    int n_accepted;
    int n_iterated;

    pair<int, int> selectRandom2Edges(int n_Node){
        int randNode1 = randi<int>(distr_param(0, n_Node - 1));
        int randNode2 = randi<int>(distr_param(0, n_Node - 1));
        while (randNode1 == randNode2) {
            randNode2 = randi<int>(distr_param(0, n_Node - 1));
        }
        pair<int,int> res = { randNode1, randNode2 };
        return res;
        
    }

    pair<Network, int> proposeNet(Network lastNet) {
        int n_node = lastNet.get_n_Node();
        pair<int, int> changeEdgeIndex = selectRandom2Edges(n_node);
        Mat<int> proposalNetStructure = lastNet.get_netStructure();
        int Y_ij = proposalNetStructure(changeEdgeIndex.first, changeEdgeIndex.second);

        proposalNetStructure(changeEdgeIndex.first, changeEdgeIndex.second) = 1 - Y_ij;
        proposalNetStructure(changeEdgeIndex.second, changeEdgeIndex.first) = 1 - Y_ij;
        Network proposalNet = Network(proposalNetStructure, false);

        pair<Network, int> res = { proposalNet, Y_ij };
        return res;
    }

    double log_r(Network lastNet, pair<Network, int> proposedNetPair) {
        //model specify 분리할 수 있으면 좋긴할듯 (어떻게?)
        //NOW: model : n_Edge
        Col<double> model_delta = { (double) proposedNetPair.first.get_n_Edge() - lastNet.get_n_Edge() }; // <-model specify
        Col<double> log_r_col = (given_param * model_delta);
        double res = log_r_col(0);
        if (proposedNetPair.second == 1) res *= -1;
        return res;
    }

    void sampler() {
        Network lastNet = MCMCSampleVec.back();
        pair<Network, int> proposedNetPair = proposeNet(lastNet);
        double log_unif_sample = log(randu());
        double log_r_val = log_r(lastNet, proposedNetPair);
        if (log_unif_sample < log_r_val) {
            //accept
            MCMCSampleVec.push_back(proposedNetPair.first);
            n_accepted++;
            n_iterated++;
        }
        else {
            //reject
            MCMCSampleVec.push_back(lastNet);
            n_iterated++;
        }
    }

    

public:
    netMCMCSampler(Col<double> param, Network initialNet) {
        given_param = param.t();
        MCMCSampleVec.push_back(initialNet);
        //model도 받도록 나중에
    }
    netMCMCSampler(Row<double> param, Network initialNet) {
        given_param = param;
        MCMCSampleVec.push_back(initialNet);
        //model도 받도록 나중에
    }

    void generateSample(int num_iter) {
        for (int i = 0; i < num_iter; i++) {
            sampler();
        }
        // cout << "MCMC done: " << n_iterated << " networks are generated." << endl;
    }

    void cutBurnIn(int n_burn_in) {
        MCMCSampleVec.erase(MCMCSampleVec.begin(), MCMCSampleVec.begin() + n_burn_in + 1);
    }

    vector<Network> getMCMCSampleVec() {
        return MCMCSampleVec;
    }

    void testOut() {
        int i = 0;
        while (i<MCMCSampleVec.size()) {
            Network printedNet = MCMCSampleVec[i];
            cout << "#" << i << endl;
            printedNet.printSummary();
            i++;
        }
        
    }
};


class optimizeParam {
private:
    vector<Col<double>> ParamSequence;
    Network observedNet;
    int n_Node;
    
    void updateNetworkInfo() {
        n_Node = observedNet.get_n_Node();
    }

    vector<Network> genSampleByMCMC(int m_Smpl, int m_burnIn) {
        //undirected graph
        Col<double> lastParam = ParamSequence.back();
        Mat<int> initialNetStructure; // isolated graph로 시작 (random으로 뿌리면 더 좋을듯)
        initialNetStructure.zeros(n_Node, n_Node);
        Network initialNet(initialNetStructure, 0);

        netMCMCSampler MCMCsampler(lastParam, initialNet);
        MCMCsampler.generateSample(m_Smpl);
        MCMCsampler.cutBurnIn(m_burnIn);
        vector<Network> MCMCSampleVec = MCMCsampler.getMCMCSampleVec();
        return MCMCSampleVec;
    }


    Col<double> genWeight(Col<double> lastParam, vector<Col<double>> MCMCSample_ModelVal) {
        Col<double> initParam = ParamSequence.front();
        Col<double> weight = zeros(MCMCSample_ModelVal.size());
        for (int i = 0; i < MCMCSample_ModelVal.size(); i++) {
            Col<double> beforeExp = (lastParam - initParam).t()* MCMCSample_ModelVal[i];
            weight[i] = exp(beforeExp(0));
        }
        double weightTotal = accu(weight);
        weight = weight / weightTotal;
        return weight;
    }


    Mat<double> invInfoCal(Col<double> lastParam, vector<Col<double>> MCMCSample_ModelVal, Col<double> weight, Col<double> wZsum) {
        Mat<double> FisherInfo = zeros(lastParam.size(), lastParam.size());
        Mat<double> wZZsum = zeros(lastParam.size(), lastParam.size());
        
        for (int i = 0; i < MCMCSample_ModelVal.size(); i++) {
            wZZsum += MCMCSample_ModelVal[i] * (MCMCSample_ModelVal[i].t()) * weight[i];
        }
        FisherInfo = wZZsum - wZsum * (wZsum.t());
        Mat<double> invFisherInfo = inv(FisherInfo);
        return invFisherInfo;
    }

    Col<double> NRupdate1Step() {
        Col<double> lastParam = ParamSequence.back();

        //MCMC
        int m_MCSample = 1000;
        vector<Network> MCMCSampleVec = genSampleByMCMC(10000, 9000);
        
        //make MCMCSample_ModelVal vector for each MCMC sample (in paper, Z_i)
        //NOW: model : n_Edge
        vector<Col<double>> MCMCSample_ModelVal; //Z_i vectors
        for (int i = 0; i < MCMCSampleVec.size(); i++) {
            Col<double> val = { (double)MCMCSampleVec[i].get_n_Edge() }; // <- model specify
            MCMCSample_ModelVal.push_back(val);
        }

        //make weight vector (in paper, w_i)
        Col<double> weight = genWeight(lastParam, MCMCSample_ModelVal);

        //Newton-Raphson
        Col<double> Observed_ModelVal = {(double) observedNet.get_n_Edge() };

        Col<double> wZsum = zeros(lastParam.size());
        for (int i = 0; i < MCMCSample_ModelVal.size(); i++) {
            wZsum += MCMCSample_ModelVal[i] * weight[i];
        }
        Mat<double> invFisherInfo = invInfoCal(lastParam, MCMCSample_ModelVal, weight, wZsum);

        
        Col<double> newParam = lastParam + invFisherInfo * ( Observed_ModelVal - wZsum );
        return newParam;
    }


public:
    optimizeParam(Col<double> initialParam, Network observed) {
        ParamSequence.push_back(initialParam);
        observedNet = observed;
        updateNetworkInfo();
    }
    
    void RunOptimize() {
        bool eq = false;
        int runNWnum = 0;
        double epsilon_thres = 0.002;
        while (!eq) {
            cout << "N-R iter" << runNWnum << endl;
            Col<double> lastParam = ParamSequence.back();
            Col<double> newParam = NRupdate1Step();
            ParamSequence.push_back(newParam);
            eq = approx_equal(lastParam, newParam, "absdiff", epsilon_thres);
            cout << "proposed: " << newParam << endl;
            runNWnum++;
        }
        cout << "optimized! iter:" << runNWnum << endl;
    }

    void testOut() {
        vector<Network> MCMCSampleVec = genSampleByMCMC(10, 8);
        int i = 0;
        while (i < MCMCSampleVec.size()) {
            Network printedNet = MCMCSampleVec[i];
            cout << "#" << i << endl;
            printedNet.printSummary();
            i++;
        }
        
    }
};


int main()
{
    Mat<int> A = {  {0,1,1,0,0,1},
                    {1,0,1,0,0,0},
                    {1,1,0,0,1,0},
                    {0,0,0,0,0,1},
                    {0,0,1,0,0,1},
                    {1,0,0,1,1,0} }
    ;

    Network netA = Network(A, false);
    netA.printSummary();


    //MCMCsampler test
    //Col<double> testParam = { 0.1 };
    //netMCMCSampler sampler(testParam, netA);
    //sampler.generateSample(10);
    //sampler.testOut();
    //sampler.cutBurnIn(8);
    //cout << "after burnin" << endl;
    //sampler.testOut();

    //Optimizer test
    Col<double> initParam = { 0.0 };
    optimizeParam OptimizerA (initParam, netA);
    OptimizerA.RunOptimize();

    //Col<double> testvec = { 1,2,3,4,5 };
    //testvec = testvec / 5;
    //cout << testvec << endl;



    return 0;
}
