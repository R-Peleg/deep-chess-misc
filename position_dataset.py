"""
Define an iterable dataset for chess position
"""
from torch.utils.data import IterableDataset
import chess.pgn


class PositionDataset(IterableDataset):
    def _game_to_position_iter(self, game: chess.pgn.Game):
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            yield board

    def __init__(self, pgn_file):
        self.pgn_file = pgn_file

    def __iter__(self):
        self.pgn_file.seek(0)
        game = chess.pgn.read_game(self.pgn_file)
        while game is not None:
            for position in self._game_to_position_iter(game):
                yield position
            game = chess.pgn.read_game(self.pgn_file)
