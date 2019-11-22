#include <iostream>

#include "board.h"
using game::Board;
using game::Move;
using game::pc_t;

using std::vector;
using std::cout;
using std::cin;
using std::cerr;
using std::endl;

int main(int argc, char **argv) {
    string fen;

    cout << "Fen (\"default\" for starting fen): ";
    cin >> fen;
    cout << endl;

    Board board;
    if (fen.compare("default") != 0) {
        board = Board(fen.c_str());
    }

    vector<Move> moves = board.generateMoves();

    while (moves.size() > 0) {
        // status
        cout << "Board:" << endl << board << endl;
        cout << "Fen: " << board.toFen() << endl;
        cout << ((FLAGS_WPLAYER(board.flags_)) ? "White" : "Black") << " to play" << endl;

        // list moves
        cout << "Moves: ";
        for (size_t i = 0; i < moves.size(); ++i) {
            cout << " " << i << ". " << moves[i];
        }
        cout << endl;

        // prompt user move
        string choiceStr;
        size_t choice;
        do {
            cout << "Choice: ";
            cin >> choiceStr;
            cout << endl;
            choice = atoi(choiceStr.c_str());
        } while (choice < 0 || choice >= moves.size());

        board.applyMove(moves[choice]);
        moves = board.generateMoves();
    }

    cout << "Game over." << endl;

    return 0;
}
