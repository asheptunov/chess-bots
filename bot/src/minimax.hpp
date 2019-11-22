#pragma once

#include <vector>
#include <string>

#include "board.h"
using game::Board;
using game::Move;
using game::pc_t;

using std::vector;
using std::string;

namespace bots {

// evaluates the board. higher score is better for the active player.
int eval(const Board &board);
// evaluates a piece. higher score is higher value.
int pcval(pc_t piece);
// performs basic minimax on a board with a depth limit and custom eval function
vector<string> minimax(int (*eval)(const Board &), const Board &board, const int depth, Move *mv, int *val, int *nodeCtr);

}  // namespace bots
