#include "Solver.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>

#include <cmath>


using namespace std;


namespace szx {

#pragma region Solver::Cli
int Solver::Cli::run(int argc, char * argv[]) {
    Log(LogSwitch::Szx::Cli) << "parse command line arguments." << endl;
    Set<String> switchSet;
    Map<String, char*> optionMap({ // use string as key to compare string contents instead of pointers.
        { InstancePathOption(), nullptr },
        { SolutionPathOption(), nullptr },
        { RandSeedOption(), nullptr },
        { TimeoutOption(), nullptr },
        { MaxIterOption(), nullptr },
        { JobNumOption(), nullptr },
        { RunIdOption(), nullptr },
        { EnvironmentPathOption(), nullptr },
        { ConfigPathOption(), nullptr },
        { LogPathOption(), nullptr }
    });

    for (int i = 1; i < argc; ++i) { // skip executable name.
        auto mapIter = optionMap.find(argv[i]);
        if (mapIter != optionMap.end()) { // option argument.
            mapIter->second = argv[++i];
        } else { // switch argument.
            switchSet.insert(argv[i]);
        }
    }

    Log(LogSwitch::Szx::Cli) << "execute commands." << endl;
    if (switchSet.find(HelpSwitch()) != switchSet.end()) {
        cout << HelpInfo() << endl;
    }

    if (switchSet.find(AuthorNameSwitch()) != switchSet.end()) {
        cout << AuthorName() << endl;
    }

    Solver::Environment env;
    env.load(optionMap);
    if (env.instPath.empty() || env.slnPath.empty()) { return -1; }

    Solver::Configuration cfg;
    cfg.load(env.cfgPath);

    Log(LogSwitch::Szx::Input) << "load instance " << env.instPath << " (seed=" << env.randSeed << ")." << endl;
    Problem::Input input;
    if (!input.load(env.instPath)) { return -1; }

    Solver solver(input, env, cfg);
    solver.solve();

    pb::Submission submission;
    submission.set_thread(to_string(env.jobNum));
    submission.set_instance(env.friendlyInstName());
    submission.set_duration(to_string(solver.timer.elapsedSeconds()) + "s");

    solver.output.save(env.slnPath, submission);
    #if SZX_DEBUG
    solver.output.save(env.solutionPathWithTime(), submission);
    solver.record();
    #endif // SZX_DEBUG

    return 0;
}
#pragma endregion Solver::Cli

#pragma region Solver::Environment
void Solver::Environment::load(const Map<String, char*> &optionMap) {
    char *str;

    str = optionMap.at(Cli::EnvironmentPathOption());
    if (str != nullptr) { loadWithoutCalibrate(str); }

    str = optionMap.at(Cli::InstancePathOption());
    if (str != nullptr) { instPath = str; }

    str = optionMap.at(Cli::SolutionPathOption());
    if (str != nullptr) { slnPath = str; }

    str = optionMap.at(Cli::RandSeedOption());
    if (str != nullptr) { randSeed = atoi(str); }

    str = optionMap.at(Cli::TimeoutOption());
    if (str != nullptr) { msTimeout = static_cast<Duration>(atof(str) * Timer::MillisecondsPerSecond); }

    str = optionMap.at(Cli::MaxIterOption());
    if (str != nullptr) { maxIter = atoi(str); }

    str = optionMap.at(Cli::JobNumOption());
    if (str != nullptr) { jobNum = atoi(str); }

    str = optionMap.at(Cli::RunIdOption());
    if (str != nullptr) { rid = str; }

    str = optionMap.at(Cli::ConfigPathOption());
    if (str != nullptr) { cfgPath = str; }

    str = optionMap.at(Cli::LogPathOption());
    if (str != nullptr) { logPath = str; }

    calibrate();
}

void Solver::Environment::load(const String &filePath) {
    loadWithoutCalibrate(filePath);
    calibrate();
}

void Solver::Environment::loadWithoutCalibrate(const String &filePath) {
    // EXTEND[szx][8]: load environment from file.
    // EXTEND[szx][8]: check file existence first.
}

void Solver::Environment::save(const String &filePath) const {
    // EXTEND[szx][8]: save environment to file.
}
void Solver::Environment::calibrate() {
    // adjust thread number.
    int threadNum = thread::hardware_concurrency();
    if ((jobNum <= 0) || (jobNum > threadNum)) { jobNum = threadNum; }

    // adjust timeout.
    msTimeout -= Environment::SaveSolutionTimeInMillisecond;
}
#pragma endregion Solver::Environment

#pragma region Solver::Configuration
void Solver::Configuration::load(const String &filePath) {
    // EXTEND[szx][5]: load configuration from file.
    // EXTEND[szx][8]: check file existence first.
}

void Solver::Configuration::save(const String &filePath) const {
    // EXTEND[szx][5]: save configuration to file.
}
#pragma endregion Solver::Configuration

#pragma region Solver
bool Solver::solve() {
    init();

    int workerNum = (max)(1, env.jobNum / cfg.threadNumPerWorker);
    cfg.threadNumPerWorker = env.jobNum / workerNum;
    List<Solution> solutions(workerNum, Solution(this));
    List<bool> success(workerNum);

    Log(LogSwitch::Szx::Framework) << "launch " << workerNum << " workers." << endl;
    List<thread> threadList;
    threadList.reserve(workerNum);
    for (int i = 0; i < workerNum; ++i) {
        // TODO[szx][2]: as *this is captured by ref, the solver should support concurrency itself, i.e., data members should be read-only or independent for each worker.
        // OPTIMIZE[szx][3]: add a list to specify a series of algorithm to be used by each threads in sequence.
        threadList.emplace_back([&, i]() { success[i] = optimize(solutions[i], i); });
    }
    for (int i = 0; i < workerNum; ++i) { threadList.at(i).join(); }

    Log(LogSwitch::Szx::Framework) << "collect best result among all workers." << endl;
    int bestIndex = -1;
    Length bestValue = Problem::MaxColorNum;
    for (int i = 0; i < workerNum; ++i) {
        if (!success[i]) { continue; }
        Log(LogSwitch::Szx::Framework) << "worker " << i << " got " << solutions[i].colorNum << endl;
        if (solutions[i].colorNum >= bestValue) { continue; }
        bestIndex = i;
        bestValue = solutions[i].colorNum;
    }

    env.rid = to_string(bestIndex);
    if (bestIndex < 0) { return false; }
    output = solutions[bestIndex];
    return true;
}

void Solver::record() const {
    #if SZX_DEBUG
    int generation = 0;

    ostringstream log;

    System::MemoryUsage mu = System::peakMemoryUsage();

    Length obj = output.colorNum;
    Length checkerObj = -1;
    bool feasible = check(checkerObj);

    // record basic information.
    log << env.friendlyLocalTime() << ","
        << env.rid << ","
        << env.instPath << ","
        << feasible << "," << (obj - checkerObj) << ","
        << obj << ","
        << timer.elapsedSeconds() << ","
        << mu.physicalMemory << "," << mu.virtualMemory << ","
        << env.randSeed << ","
        << cfg.toBriefStr() << ","
        << generation << "," << iteration << ",";

    // record solution vector.
    for (auto n = output.nodecolors().begin(); n != output.nodecolors().end(); ++n) {
        log << *n << " ";
    }
    log << endl;

    // append all text atomically.
    static mutex logFileMutex;
    lock_guard<mutex> logFileGuard(logFileMutex);

    ofstream logFile(env.logPath, ios::app);
    logFile.seekp(0, ios::end);
    if (logFile.tellp() <= 0) {
        logFile << "Time,ID,Instance,Feasible,ObjMatch,Color,Duration,PhysMem,VirtMem,RandSeed,Config,Generation,Iteration,Solution" << endl;
    }
    logFile << log.str();
    logFile.close();
    #endif // SZX_DEBUG
}

bool Solver::check(Length &checkerObj) const {
    #if SZX_DEBUG
    enum CheckerFlag {
        IoError = 0x0,
        FormatError = 0x1,
        ColorConflictError = 0x2
    };

    checkerObj = System::exec("Checker.exe " + env.instPath + " " + env.solutionPathWithTime());
    if (checkerObj > 0) { return true; }
    checkerObj = ~checkerObj;
    if (checkerObj == CheckerFlag::IoError) { Log(LogSwitch::Checker) << "IoError." << endl; }
    if (checkerObj & CheckerFlag::FormatError) { Log(LogSwitch::Checker) << "FormatError." << endl; }
    if (checkerObj & CheckerFlag::ColorConflictError) { Log(LogSwitch::Checker) << "ColorConflictError." << endl; }
    return false;
    #else
    checkerObj = 0;
    return true;
    #endif // SZX_DEBUG
}

void Solver::init() {
    ID nodeNum = input.graph().nodenum();

    aux.adjList.resize(nodeNum);
    for (auto e = input.graph().edges().begin(); e != input.graph().edges().end(); ++e) {
        // assume there is no duplicated edge.
        aux.adjList.at(e->src()).push_back(e->dst());
        aux.adjList.at(e->dst()).push_back(e->src());
    }

}

bool Solver::optimize(Solution &sln, ID workerId) {
    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " starts." << endl;

    ID nodeNum = input.graph().nodenum();
    ID colorNum = input.colornum();

    // reset solution state.
    bool status = true;
    auto &nodeColors(*sln.mutable_nodecolors());
    nodeColors.Resize(nodeNum, Problem::InvalidId);

    // TODO[0]: replace the following random assignment with your own algorithm.
	Grapthassess grapthassess(colorNum, aux.adjList, rand) ;//定义图的结构的相关简化数据结构【仇人表、颜色表...】

	//Timer timer_tabu(std::chrono::milliseconds(env.msTimeout));
	//tabusearch(grapthassess, timer_tabu);//对其进行搜索和调整（这里是禁忌算法）.

	hybird_evoluation(grapthassess, 8, 10*1000);//进化算法

    for (ID n = 0; (n < nodeNum); ++n) {//输出结果
        //nodeColors[n] = rand.pick(colorNum);
		nodeColors[n] = grapthassess.getcolor(n);
    }
	



    sln.colorNum = input.colornum(); // record obj.

    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " ends." << endl;
    return status;
}
void Solver::tabusearch(Grapthassess & grapthassess,const Timer& timer)
{
	ID nodeNum = input.graph().nodenum();
	ID colorNum = input.colornum();
	List<List<TabuTime>> tabulist(nodeNum, vector<TabuTime>(colorNum, 0));
	/*
	while (!timer.isTimeOut() && grapthassess.getconflictNum()) {//贪心.
		ObjValue bestchange = numeric_limits<ObjValue>::max();
		ID bestnode = -1, bestcolor = -1;
		Quantity bestcount = 0;
		for (int i = 0; i < grapthassess.confilictNodes.size(); ++i) {//搜索所有冲突节点试图将它们改变颜色
			ID nodeid = grapthassess.confilictNodes.idList[i];
			for (ID color = 0; color < colorNum; ++color){
				if (color == grapthassess.getcolor(nodeid))
					continue;

					if (grapthassess.objchange(nodeid, color) < bestchange) {
						//获取使得FS减少最大的点和颜色
						bestchange = grapthassess.objchange(nodeid, color);
						bestnode = nodeid;
						bestcolor = color;
						bestcount = 1;
					}
					else if (grapthassess.objchange(nodeid, color) == bestchange
						&& rand.isPicked(1, ++bestcount)) {
						//倘若有多个相同的减少量，则随机获取任一一对
							bestnode = nodeid;
							bestcolor = color;
					}
			}
		}

		cout << "bestnode= " << bestnode << " bestcolor= " << bestcolor << " ";
		grapthassess.change2newcolor(bestnode, bestcolor);//变色
		cout << "FS= " << grapthassess.getconflictNum() << endl;
	}
	*/
	
	TabuTime t = 0;
	ObjValue bestFS = grapthassess.getconflictNum();
	
	while (!timer.isTimeOut() && grapthassess.getconflictNum()) {//禁忌.
		ObjValue bestchange = numeric_limits<ObjValue>::max();
		ID bestnode = -1, bestcolor = -1;
		Quantity bestcount = 0;
		for (int i = 0; i < grapthassess.confilictNodes.size(); ++i) {//搜索所有冲突节点试图将它们改变颜色
			ID nodeid = grapthassess.confilictNodes.idList[i];
			for (ID color = 0; color < colorNum; ++color) {
				if (color == grapthassess.getcolor(nodeid))//必须要变色
					continue;
				if (tabulist[nodeid][color] <= t
					|| grapthassess.getconflictNum() + grapthassess.objchange(nodeid, color) < bestFS) {
					//没有禁忌 + 解禁策略

					if (grapthassess.objchange(nodeid, color) < bestchange) {
						//获取使得FS减少最大的点和颜色
						bestchange = grapthassess.objchange(nodeid, color);
						bestnode = nodeid;
						bestcolor = color;
						bestcount = 1;
					}
					else if (grapthassess.objchange(nodeid, color) == bestchange
						&& rand.isPicked(1, ++bestcount)) {
						//倘若有多个相同的减少量，则随机获取任一一对
							bestnode = nodeid;
							bestcolor = color;
					}
				}


			}
				
		}
		
		//cout << "bestnode = " << bestnode << " bestcolor = " << bestcolor <<" t = "<<t;
		ID oldcolor = grapthassess.getcolor(bestnode);
		//tabulist[bestnode][bestcolor] = get_tabustep(t) + t;//加入禁忌
		tabulist[bestnode][oldcolor] = grapthassess.getconflictNum() + get_tabustep(t) + t++;//加入禁忌

		grapthassess.change2newcolor(bestnode, bestcolor);//点变色
		/*if (bestFS > grapthassess.getconflictNum() && bestFS <= 20)
			cout << "bestFS = " << grapthassess.getconflictNum() <<" t = "<< t <<endl;*/
		bestFS = min(bestFS, grapthassess.getconflictNum());//更新最优解
		//cout << " FS= " << grapthassess.getconflictNum() << endl;
	}
	return;
}
void Solver::crossover_operator(const Grapthassess & s0, const Grapthassess & s1, Grapthassess & mixeds)
{
	ID nodeNum = input.graph().nodenum();
	ID colorNum = input.colornum();
	vector<bool>is_select(nodeNum, false);//点是否被选过
	vector<vector<Quantity>>s_color_cnt(2, vector<int>(colorNum, 0));

	vector<int>mixeds_color(nodeNum, -1);//保存解.

	for (ID nodeid = 0; nodeid < nodeNum; ++nodeid) {
		++s_color_cnt[0][s0.getcolor(nodeid)];
		++s_color_cnt[1][s1.getcolor(nodeid)];
	}
	bool select_id = rand.isPicked(1, 2);
	for (ID colorid = 0; colorid < colorNum; ++colorid) {//每次给mixeds涂一个颜色.
		ID select_color = -1;
		Quantity maxcnt = -1;
		Quantity cnt = 0;
		for (ID i = 0; i < colorNum; ++i) {//选择哪个颜色块下移
			if (s_color_cnt[select_id].at(i) > maxcnt) {
				maxcnt = s_color_cnt[select_id].at(i);
				select_color = i;
				cnt = 1;
			}
			else if (s_color_cnt[select_id].at(i) == maxcnt
				&& rand.isPicked(1, ++cnt)) {
				select_color = i;
			}
		}
		const Grapthassess & s = select_id ? s1 : s0;
		for (ID nodeid = 0; nodeid < nodeNum; ++nodeid)
			if (s.getcolor(nodeid) == select_color
				&& is_select[nodeid] == false) {//寻找这个解的颜色块的点 要求：没被选过

				mixeds_color[nodeid] = colorid;//给这个点涂上mixeds的颜色
				is_select[nodeid] = true;
				--s_color_cnt[0][s0.getcolor(nodeid)];
				--s_color_cnt[1][s1.getcolor(nodeid)];
			}

		select_id = !select_id;//换一方选择.
	}

	for (ID nodeid = 0; nodeid < nodeNum; ++nodeid)
		if (is_select[nodeid] == false)//最后没被选的点随机染色即可
			mixeds_color[nodeid] = rand.pick(colorNum);

	mixeds.setcolor(mixeds_color);
	return;
}
void Solver::hybird_evoluation(Grapthassess & grapthassess, Quantity populations, Duration tabu_time)//tabu_time is millonsecond
{
	double one_tabu_time = tabu_time/1000.0;//second
	one_tabu_time = 1;//to debug and test.
	ID nodeNum = input.graph().nodenum();
	ID colorNum = input.colornum();
	vector<Grapthassess>ans(populations, grapthassess);

	vector<ID>nodecolor(nodeNum, -1);
	for (int i = 0; i < populations; ++i) {
		for (auto& n : nodecolor)
			n = rand.pick(colorNum);
		ans[i].setcolor(nodecolor);
		//Timer timer_tabu(std::chrono::milliseconds(timer.toMillisecond(one_tabu_time)));
		//tabusearch(ans[i], timer_tabu);
		//if (ans[i].getconflictNum() == 0) {
		//	for (ID nodeid = 0; nodeid < nodeNum; ++nodeid)//进行族群替换
		//		nodecolor[nodeid] = ans[i].getcolor(nodeid);
		//	grapthassess.setcolor(nodecolor);
		//	return;
		//}
	}
	while (!timer.isTimeOut() && grapthassess.getconflictNum()) {
		ID id1 = 0, id2 = 0;
		Quantity cnt = 0;
		ObjValue best = numeric_limits<ObjValue>::max();
		for (ID i = 0; i < populations; ++i) {
			if (best > ans[i].getconflictNum()) {
				best = ans[i].getconflictNum();
				id1 = i;
				cnt = 1;
			}
			else if (best == ans[i].getconflictNum()
				&& rand.isPicked(1, ++cnt)) {
				id1 = i;
			}
		}
		cout << "bestFS = " << ans[id1].getconflictNum() << endl;
		for (ID i = 0; i < populations; ++i) {
			if (i == id1)continue;
			if (best > ans[i].getconflictNum()) {
				best = ans[i].getconflictNum();
				id2 = i;
				cnt = 1;
			}
			else if (best == ans[i].getconflictNum()
				&& rand.isPicked(1, ++cnt)) {
				id2 = i;
			}
		}
		crossover_operator(ans[id1], ans[id2], grapthassess);//交叉算符
		Timer timer_tabu(std::chrono::milliseconds(timer.toMillisecond(one_tabu_time)));
		tabusearch(grapthassess, timer_tabu);
		if (grapthassess.getconflictNum() == 0)break;

		if (grapthassess.getconflictNum() > ans[id1].getconflictNum() 
			&& grapthassess.getconflictNum() > ans[id2].getconflictNum())//没有变得更好就不更新族群。
			continue;

		ID id3 = 0;//找最差的种子.
		cnt = 0;
		best = -1;
		for (ID i = 0; i < populations; ++i) {
			//if (i == id1)continue;
			//if (i == id2)continue;
			if (best < ans[i].getconflictNum()) {
				best = ans[i].getconflictNum();
				id3 = i;
				cnt = 1;
			}
			else if (best == ans[i].getconflictNum()
				&& rand.isPicked(1, ++cnt)) {
				id3 = i;
			}
		}

		for (ID nodeid = 0; nodeid < nodeNum; ++nodeid)//进行族群替换
			nodecolor[nodeid] = grapthassess.getcolor(nodeid);
		ans[id3].setcolor(nodecolor);
	}
	return;

}
#pragma endregion Solver

bool Solver::Grapthassess::isconflictnode(ID nodeid) const
{
	ID nodecolor = getcolor(nodeid);
	return conflictTable[nodeid][nodecolor] != 0;
}

ObjValue Solver::Grapthassess::objchange(ID nodeid, ID newcolor)const
{
	ID oldcolor = nodeColor[nodeid];
	return conflictTable[nodeid][newcolor]-conflictTable[nodeid][oldcolor];
}

void Solver::Grapthassess::change2newcolor(ID nodeid, ID newcolor)
{
	FS += objchange(nodeid, newcolor);

	ID oldcolor = nodeColor[nodeid];
	for (auto borderid : adjList[nodeid]) {
		--conflictTable[borderid][oldcolor];
		++conflictTable[borderid][newcolor];
		if (isconflictnode(borderid))
			confilictNodes.addid(borderid);
		else
			confilictNodes.deleteid(borderid);
	}
	nodeColor[nodeid] = newcolor;
	if (!isconflictnode(nodeid))//要判断node本身是否变为非冲突。
		confilictNodes.deleteid(nodeid);
}

void Solver::Grapthassess::randomcoloring()
{
	//List<List<Quantity>> conflictTable;
	//List<ID> nodeColor;
	nodeColor.resize(nodeNum);
	for (auto &node : nodeColor)
		node = rand.pick(colorNum);

	return;

}

void Solver::Grapthassess::init()
{
	confilictNodes.clear();
	conflictTable.clear();
	conflictTable.resize(nodeNum, List<Quantity>(colorNum, 0));

	FS = 0;
	for (int nodeID = 0; nodeID < nodeNum; ++nodeID)
		for (auto borderID : adjList[nodeID]) {
			++conflictTable[nodeID][getcolor(borderID)];
			if (getcolor(borderID) == getcolor(nodeID)) {
				++FS;
				confilictNodes.addid(nodeID);
			}
		}

	FS /= 2;
}

}
