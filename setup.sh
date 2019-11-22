usage() {
    echo "usage: setup.sh (-o|--override)?"
}

setup() {
    if ! [[ -d bot/lib ]] ; then
        mkdir bot/lib
    fi
    if ! [[ -d bot/lib/chess ]] ; then
        mkdir bot/lib
    fi
    if ! [[ -d bot/lib/chess/include ]] ; then
        mkdir bot/lib
    fi
    if ! [[ -d bot/lib/chess/lib ]] ; then
        mkdir bot/lib
    fi
}

setup

here=$(pwd)
chesslib=$(jq -r ".dependencies.chess_engine.remote.url" build.json)
chesslib_branch=$(jq -r ".dependencies.chess_engine.remote.branch" build.json)

# read arguments
if [ "$1" != "" ] ;
    then
        case "$1" in
            -o | --override )
                chesslib=$(jq -r ".dependencies.chess_engine.local.dir" build.json)
                ;;
            * )
                usage
                exit 1
                ;;
        esac
fi

if [[ $chesslib =~ https?://.+\.git$ ]] ;
    then
        # REMOTE git url; clone, cd, build, extract, and cleanup
        git clone $chesslib
        cd chess-engine
        git checkout $chesslib_branch
        sh setup.sh >/dev/null
        cp chess/include/*.h $here/bot/lib/chess/include/
        cp chess/bin/libchess.a $here/bot/lib/chess/lib/libchess.a
        cd ..
        rm -rf chess-engine
    else
        # LOCAL (override) directory; cd, build, extract, cleanup
        cd $chesslib
        sh setup.sh >/dev/null
        cp chess/include/*.h $here/bot/lib/chess/include/
        cp chess/bin/libchess.a $here/bot/lib/chess/lib/libchess.a
        cd $here
fi

echo "Done."

exit 0
