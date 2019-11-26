////////////////////////////////
/// usage : 1.	the entry to the algorithm for the guillotine cut problem.
/// 
/// note  : 1.	
////////////////////////////////

#ifndef SMART_SZX_GRAPH_COLORING_SOLVER_H
#define SMART_SZX_GRAPH_COLORING_SOLVER_H


#include "Config.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <sstream>
#include <thread>

#include "Common.h"
#include "Utility.h"
#include "LogSwitch.h"
#include "Problem.h"


namespace szx {

class Solver {
    #pragma region Type
public:
    // commmand line interface.
    struct Cli {
        static constexpr int MaxArgLen = 256;
        static constexpr int MaxArgNum = 32;

        static String InstancePathOption() { return "-p"; }
        static String SolutionPathOption() { return "-o"; }
        static String RandSeedOption() { return "-s"; }
        static String TimeoutOption() { return "-t"; }
        static String MaxIterOption() { return "-i"; }
        static String JobNumOption() { return "-j"; }
        static String RunIdOption() { return "-rid"; }
        static String EnvironmentPathOption() { return "-env"; }
        static String ConfigPathOption() { return "-cfg"; }
        static String LogPathOption() { return "-log"; }

        static String AuthorNameSwitch() { return "-name"; }
        static String HelpSwitch() { return "-h"; }

        static String AuthorName() { return "szx"; }
        static String HelpInfo() {
            return "Pattern (args can be in any order):\n"
                "  exe (-p path) (-o path) [-s int] [-t seconds] [-name]\n"
                "      [-iter int] [-j int] [-id string] [-h]\n"
                "      [-env path] [-cfg path] [-log path]\n"
                "Switches:\n"
                "  -name  return the identifier of the authors.\n"
                "  -h     print help information.\n"
                "Options:\n"
                "  -p     input instance file path.\n"
                "  -o     output solution file path.\n"
                "  -s     rand seed for the solver.\n"
                "  -t     max running time of the solver.\n"
                "  -i     max iteration of the solver.\n"
                "  -j     max number of working solvers at the same time.\n"
                "  -rid   distinguish different runs in log file and output.\n"
                "  -env   environment file path.\n"
                "  -cfg   configuration file path.\n"
                "  -log   activate logging and specify log file path.\n"
                "Note:\n"
                "  0. in pattern, () is non-optional group, [] is optional group\n"
                "     when -env option is not given.\n"
                "  1. an environment file contains information of all options.\n"
                "     explicit options get higher priority than this.\n"
                "  2. reaching either timeout or iter will stop the solver.\n"
                "     if you specify neither of them, the solver will be running\n"
                "     for a long time. so you should set at least one of them.\n"
                "  3. the solver will still try to generate an initial solution\n"
                "     even if the timeout or max iteration is 0. but the solution\n"
                "     is not guaranteed to be feasible.\n";
        }

        // a dummy main function.
        static int run(int argc, char *argv[]);
    };

    // controls the I/O data format, exported contents and general usage of the solver.
    struct Configuration {
        enum Algorithm { Greedy, TreeSearch, DynamicProgramming, LocalSearch, Genetic, MathematicallProgramming };


        Configuration() {}

        void load(const String &filePath);
        void save(const String &filePath) const;


        String toBriefStr() const {
            String threadNum(std::to_string(threadNumPerWorker));
            std::ostringstream oss;
            oss << "alg=" << alg
                << ";job=" << threadNum;
            return oss.str();
        }


        Algorithm alg = Configuration::Algorithm::Greedy; // OPTIMIZE[szx][3]: make it a list to specify a series of algorithms to be used by each threads in sequence.
        int threadNumPerWorker = (std::min)(1, static_cast<int>(std::thread::hardware_concurrency()));
    };

    // describe the requirements to the input and output data interface.
    struct Environment {
        static constexpr int DefaultTimeout = (1 << 30);
        static constexpr int DefaultMaxIter = (1 << 30);
        static constexpr int DefaultJobNum = 0;
        // preserved time for IO in the total given time.
        static constexpr int SaveSolutionTimeInMillisecond = 1000;

        static constexpr Duration RapidModeTimeoutThreshold = 600 * static_cast<Duration>(Timer::MillisecondsPerSecond);

        static String DefaultInstanceDir() { return "Instance/"; }
        static String DefaultSolutionDir() { return "Solution/"; }
        static String DefaultVisualizationDir() { return "Visualization/"; }
        static String DefaultEnvPath() { return "env.csv"; }
        static String DefaultCfgPath() { return "cfg.csv"; }
        static String DefaultLogPath() { return "log.csv"; }

        Environment(const String &instancePath, const String &solutionPath,
            int randomSeed = Random::generateSeed(), double timeoutInSecond = DefaultTimeout,
            Iteration maxIteration = DefaultMaxIter, int jobNumber = DefaultJobNum, String runId = "",
            const String &cfgFilePath = DefaultCfgPath(), const String &logFilePath = DefaultLogPath())
            : instPath(instancePath), slnPath(solutionPath), randSeed(randomSeed),
            msTimeout(static_cast<Duration>(timeoutInSecond * Timer::MillisecondsPerSecond)), maxIter(maxIteration),
            jobNum(jobNumber), rid(runId), cfgPath(cfgFilePath), logPath(logFilePath), localTime(Timer::getTightLocalTime()) {}
        Environment() : Environment("", "") {}

        void load(const Map<String, char*> &optionMap);
        void load(const String &filePath);
        void loadWithoutCalibrate(const String &filePath);
        void save(const String &filePath) const;

        void calibrate(); // adjust job number and timeout to fit the platform.

        String solutionPathWithTime() const { return slnPath + "." + localTime; }

        String visualizPath() const { return DefaultVisualizationDir() + friendlyInstName() + "." + localTime + ".html"; }
        template<typename T>
        String visualizPath(const T &msg) const { return DefaultVisualizationDir() + friendlyInstName() + "." + localTime + "." + std::to_string(msg) + ".html"; }
        String friendlyInstName() const { // friendly to file system (without special char).
            auto pos = instPath.find_last_of('/');
            return (pos == String::npos) ? instPath : instPath.substr(pos + 1);
        }
        String friendlyLocalTime() const { // friendly to human.
            return localTime.substr(0, 4) + "-" + localTime.substr(4, 2) + "-" + localTime.substr(6, 2)
                + "_" + localTime.substr(8, 2) + ":" + localTime.substr(10, 2) + ":" + localTime.substr(12, 2);
        }

        // essential information.
        String instPath;
        String slnPath;
        int randSeed;
        Duration msTimeout;

        // optional information. highly recommended to set in benchmark.
        Iteration maxIter;
        int jobNum; // number of solvers working at the same time.
        String rid; // the id of each run.
        String cfgPath;
        String logPath;

        // auto-generated data.
        String localTime;
    };

    struct Solution : public Problem::Output { // cutting patterns.
        Solution(Solver *pSolver = nullptr) : solver(pSolver) {}

        Solver *solver;
    };
    #pragma endregion Type

    #pragma region Constant
public:
    #pragma endregion Constant

    #pragma region Constructor
public:
    Solver(const Problem::Input &inputData, const Environment &environment, const Configuration &config)
        : input(inputData), env(environment), cfg(config), rand(environment.randSeed),
        timer(std::chrono::milliseconds(environment.msTimeout)), iteration(1) {}
    #pragma endregion Constructor

    #pragma region Method
public:
    bool solve(); // return true if exit normally. solve by multiple workers together.
    bool check(Length &obj) const;
    void record() const; // save running log.

protected:
    void init();
    bool optimize(Solution &sln, ID workerId = 0); // optimize by a single worker.
	
	class Grapthassess {//��Ҫ���ڳ��˱������������.
	public:
		Grapthassess(Quantity colornum, const List<List<ID>>& adjList, Random& rand) 
			:adjList(adjList) , colorNum(colornum),rand(rand),confilictNodes(adjList.size()),nodeNum(adjList.size()){

			randomcoloring();
			init();
		};
		void setcolor(const List<ID>& nodeColor) {//���ò�ͬ��ɫ.���ܸı���ɫ��Ŀ
			this->nodeColor = nodeColor;
			init();
		};
		ID getcolor(ID nodeid) const{ return nodeColor[nodeid]; }//��ȡnodeid����ɫ
		bool isconflictnode(ID nodeid)const;//��ѯnodeid�Ƿ����г�ͻ�Ľڵ�
		ObjValue objchange(ID nodeid, ID newcolor)const;//�����л���ɫ�ĳ�ͻ�����仯�� FS+=objchange(nodeid,newcolor). 
		ObjValue getconflictNum() const{ return FS; }//��ȡĿǰ��ͻ�ıߵ���Ŀ
		void change2newcolor(ID nodeid, ID newcolor);//��һ������һ������ɫ

		ZeroBasedUnrepeteList<ID> confilictNodes;//�洢��ͻ�Ľڵ�����ݽṹ
	protected:
		void randomcoloring();//��ʼ���������ÿ���ڵ�Ⱦɫ
		void init();
		const List<List<ID>>& adjList;//ͼ���ڽӱ�
		Random& rand;//ȫ��ܵ��������������

		List<List<Quantity>> conflictTable;//��ͻ��
		List<ID> nodeColor;//ÿ���ڵ����ɫ

		Quantity colorNum;//����ɫ��Ŀ
		Quantity nodeNum;//�ܽڵ���Ŀ
		ObjValue FS;//��ͻ�ߵ���Ŀ
	};

	void tabusearch(Grapthassess & grapthassess,const Timer& timer);
	TabuTime get_tabustep(TabuTime t) { return rand.pick(10); }
	void crossover_operator(const Grapthassess& s1, const Grapthassess& s2, Grapthassess& mixeds);
	void hybird_evoluation(Grapthassess & grapthassess,Quantity populations, Duration tabu_time);
    #pragma endregion Method

    #pragma region Field
public:
    Problem::Input input;
    Problem::Output output;

    struct { // auxiliary data for solver.
        List<List<ID>> adjList; // adjMat[i][j] is the j_th adjacent node of node i.
    } aux;

    Environment env;
    Configuration cfg;

    Random rand; // all random number in Solver must be generated by this.
    Timer timer; // the solve() should return before it is timeout.
    Iteration iteration;
    #pragma endregion Field
}; // Solver 

}


#endif // SMART_SZX_GRAPH_COLORING_SOLVER_H
