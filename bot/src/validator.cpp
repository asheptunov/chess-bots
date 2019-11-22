#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <ctime>

#include "minimax.hpp"

using std::vector;
using std::string;
using std::sort;
using std::clock;
using std::cout;
using std::endl;

using bots::minimax;
using bots::eval;

// usage message
void usage();

int main(int argc, char **argv) {
    Board board = Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
    Move mv;
    int val;

    if (argc != 2) {
        usage();
    }
    const int depth = atoi(argv[1]);
    int nodeCtr = 0;

    const int start = clock();
    vector<string> paths = minimax(eval, board, depth, &mv, &val, &nodeCtr);
    const int end = clock();

    const double elapsed = (end - start) / (double) CLOCKS_PER_SEC;

    // construct string of paths
    string ret;
    sort(paths.begin(), paths.end());
    for (const string &path : paths) {
        ret.append("\n").append(path);
    }
    cout << "_Paths: " << ret << endl;
    cout << "_Touched " << nodeCtr << " nodes; best move: " << mv << "; val: " << val << "." << endl;
    cout << "_Finished in " << elapsed << " seconds; touched " << (nodeCtr / elapsed) << " nodes per second." << endl;

    return 0;
}

void usage() {
    cout << "Usage: validator ${depth}" << endl;
    exit(1);
}
