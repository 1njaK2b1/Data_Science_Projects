package game2048logic;

import game2048rendering.Board;
import game2048rendering.Side;
import game2048rendering.Tile;

import java.util.Formatter;

public class Model {
    private final Board board;
    private int score;

    public static final int MAX_PIECE = 2048;

    public Model(int size) {
        board = new Board(size);
        score = 0;
    }
    public Model(int[][] rawValues, int score) {
        board = new Board(rawValues);
        this.score = score;
    }

    public Tile tile(int x, int y) {
        return board.tile(x, y);
    }

    public int size() {
        return board.size();
    }

    public int score() {
        return score;
    }


    public void clear() {
        score = 0;
        board.clear();
    }

    public void addTile(Tile tile) {
        board.addTile(tile);
    }

   
    public boolean gameOver() {
        return maxTileExists() || !atLeastOneMoveExists();
    }

    public Board getBoard() {
        return board;
    }

    
    public boolean emptySpaceExists() {
        for (int x = 0; x < board.size(); x++) {
            for (int y = 0; y < board.size(); y++) {
                if (tile(x, y) == null) {
                    return true;
                }
            }
        }
        return false;
    }


    public boolean maxTileExists() {
        for (int x = 0; x < board.size(); x++) {
            for (int y = 0; y < board.size(); y++) {
                if (tile(x, y) != null && tile(x, y).value() == MAX_PIECE) {
                    return true;
                }
            }
        }
        return false;
    }

    
    public boolean atLeastOneMoveExists() {
        for (int x = 0; x < board.size(); x++) {
            for (int y = 0; y < board.size(); y++) {
                if (tile(x, y) == null) {
                    return true; 
                }
            }
        }


        for (int x = 0; x < board.size(); x++) {
            for (int y = 0; y < board.size(); y++) {
                Tile currentTile = tile(x, y);

                if (currentTile != null
                        && ((x > 0
                        && currentTile.value() == tile(x - 1, y).value())
                        || (x < board.size() - 1
                        && currentTile.value() == tile(x + 1, y).value())
                        || (y > 0 && currentTile.value() == tile(x, y - 1).value())
                        || (y < board.size() - 1
                        && currentTile.value() == tile(x, y + 1).value()))) {
                    return true;
                }
            }
        }
        return false;
    }

    public void moveTileUpAsFarAsPossible(int x, int y) {
        Tile currTile = board.tile(x, y);
        int myValue = currTile.value();
        int targetY = y;
        while (targetY < board.size() - 1
                && targetY >= 0
                && (board.tile(x, targetY + 1) == null)) {
            targetY++;
        }

        if (targetY < board.size() - 1 && targetY >= 0) {
            Tile tileAbove = board.tile(x, targetY + 1);
            if (tileAbove != null
                    && tileAbove.value() == myValue
                    && !tileAbove.wasMerged()
                    && !currTile.wasMerged()) {
                targetY++;
                score += currTile.value() * 2;
            }
        }
        board.move(x, targetY, currTile);
    }

    public void tiltColumn(int x) {
        for (int y = board.size() - 2; y >= 0; y--) {
            if (board.tile(x, y) != null
                    && (board.tile(x, y + 1) == null
                    || (board.tile(x, y + 1).value() == board.tile(x, y).value()
                    && !board.tile(x, y).wasMerged()
                    && !board.tile(x, y + 1).wasMerged()))) {
                moveTileUpAsFarAsPossible(x, y);
            }
        }
    }

    public void tilt(Side side) {
        board.setViewingPerspective(side);
        for (int x = board.size() - 1; x >= 0; x--) {
            tiltColumn(x);
        }
        board.setViewingPerspective(Side.NORTH);
    }
    
    public void tiltWrapper(Side side) {
        board.resetMerged();
        tilt(side);
    }

    @Override
    public String toString() {
        Formatter out = new Formatter();
        out.format("%n[%n");
        for (int y = size() - 1; y >= 0; y -= 1) {
            for (int x = 0; x < size(); x += 1) {
                if (tile(x, y) == null) {
                    out.format("|    ");
                } else {
                    out.format("|%4d", tile(x, y).value());
                }
            }
            out.format("|%n");
        }
        String over = gameOver() ? "over" : "not over";
        out.format("] %d (game is %s) %n", score(), over);
        return out.toString();
    }

    @Override
    public boolean equals(Object o) {
        return (o instanceof Model m) && this.toString().equals(m.toString());
    }

    @Override
    public int hashCode() {
        return toString().hashCode();
    }
}
