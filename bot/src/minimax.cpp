#include <vector>
#include <string>

#include "minimax.hpp"

using std::vector;
using std::string;

namespace bots {

vector<string> minimax(int (*eval)(const Board &), const Board &board, const int depth, Move *mv, int *val, int *nodeCtr) {
    vector<string> ret;

    // base case 1
    if (depth <= 0) {
        *val = eval(board);
        *nodeCtr = *nodeCtr + 1;
        return ret;
    }

    vector<Move> moves = board.generateMoves();

    // cout << moves.size() << endl;

    // base case 2
    if (moves.size() == 0) {
        // active player didn't win
        // super mega bad
        *val = -100000 - depth;  // 100k
        *nodeCtr = *nodeCtr + 1;
        return ret;
    }

    // cout << moves.size() << endl;

    for (const Move &m : moves) {
        // cout << m << endl;
        Board bFut(board);
        Move mFut;
        int vFut;

        bFut.applyMove(m);
        vector<string> res = minimax(eval, bFut, depth - 1, &mFut, &vFut, nodeCtr);

        // build str rep of explored paths
        if (res.size() == 0) {
            ret.push_back(m.algNot());
        } else {
            for (size_t i = 0; i < res.size(); ++i) {
                ret.push_back(m.algNot() + ">" + res[i]);
            }
        }

        vFut = -vFut;  // good for opp is bad for us; vice versa
        if (vFut > *val) {
            *val = vFut;
            *mv = m;
        }
    }

    return ret;
}

// evals the board as the sum of the values of the active player's pieces
int eval(const Board &board) {
    int ret = 0;
    int player = FLAGS_BPLAYER(board.flags_);  // 1 if black, 0 if white

    for (int rk = 0; rk < 8; ++rk) {
        for (int offs = 0; offs < 8; ++offs) {
            pc_t pc = (board.ranks_[rk] >> (offs * 4)) & 0xf;
            if (pc / 6 == player) {  // rate all of my pieces
                ret += pcval(pc);
            } else {
                ret -= pcval(pc);
            }
        }
    }

    return ret;
}

// returns the numerical value of a piece
int pcval(pc_t pc) {
    switch (pc / 6) {
        case WPAWN:
            return 1;
        case WKNIGHT:
        case WBISHOP:
            return 3;
        case WROOK:
            return 5;
        case WQUEEN:
            return 9;
        default:
            return 0;
    }
}

}  // namespace bots
