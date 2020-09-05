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

        int get_n_Node() {
            return n_Node;
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
        Col<double> model_delta = { (double) proposedNetPair.first.get_n_Node() - lastNet.get_n_Node() };
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

    void generate_sample(int num_iter) {
        for (int i = 0; i < num_iter; i++) {
            sampler();
        }
        cout << "MCMC done" << endl;
    }

    void testOut() {
        for (int i = 0; i < 10; i++) {
            Network printedNet = MCMCSampleVec[i];
            printedNet.printSummary();

        }
        
    }
};



int main()
{
    Mat<int> A = { {0,1,1},
                    {1,0,1},
                    {1,1,0} };

    Network netA = Network(A, false);
    netA.printSummary();


    Col<double> testParam = { 0.1 };
    netMCMCSampler sampler(testParam, netA);

    sampler.generate_sample(10);
    sampler.testOut();

    return 0;
}

