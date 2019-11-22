CHESSLIB_HDR = bot/lib/chess/include
CHESSLIB_LIB = bot/lib/chess/lib

CXX = g++
CXXFLAGS = -g -Wall -Wextra -pthread -std=c++11

all: viewer validator

clean:
	rm -rf viewer.o viewer validator.o validator

viewer.o: bot/src/viewer.cpp $(CHESSLIB_HDR) $(CHESSLIB_LIB)
	$(CXX) $(CXXFLAGS) -I $(CHESSLIB_HDR) -c -o $@ $<

viewer: viewer.o
	$(CXX) $(CXXFLAGS) -L $(CHESSLIB_LIB) -lchess $^ -o $@

minimax.o: bot/src/minimax.cpp $(CHESSLIB_HDR) $(CHESSLIB_LIB)
	$(CXX) $(CXXFLAGS) -I $(CHESSLIB_HDR) -c -o $@ $<

validator.o: bot/src/validator.cpp $(CHESSLIB_HDR) $(CHESSLIB_LIB)
	$(CXX) $(CXXFLAGS) -I $(CHESSLIB_HDR) -c -o $@ $<

validator: validator.o minimax.o
	$(CXX) $(CXXFLAGS) -L $(CHESSLIB_LIB) -lchess $^ -o $@
