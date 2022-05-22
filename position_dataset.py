"""
Define an iterable dataset for chess position
"""
from torch.utils.data import IterableDataset
import torch.nn.functional as F
import chess.pgn
import chess
import torch
import random


def position_to_tensor(position: chess.Board):
    """
    Encode a position as a 8 * 8 * 12 tensor (heigth * width * colot * piecetype)
    piecetype encoding is pnbrqkPNBRQK. tensor index [0, 0] is A1, [0, 7] is H1.
    """
    def encode_piece(piece):
        return piece.color * 6 + piece.piece_type - 1
    res = torch.zeros(64, 12)
    for sq in chess.SQUARES:
        p = position.piece_at(sq)
        if p is None:
            continue
        res[sq, encode_piece(p)] = 1
    return torch.reshape(res, (8, 8, 12))

class PositionDataset(IterableDataset):
    def _game_to_position_iter(self, game: chess.pgn.Game):
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            yield {
                'game': game,
                'position': board,
                'last_move': move
            }

    def __init__(self, pgn_file, skip_games=0, limit_games=None, take_position_prob=1):
        self.pgn_file = pgn_file
        self.skip_games = skip_games
        self.limit_games = limit_games
        self.take_position_prob = take_position_prob

    def __iter__(self):
        self.pgn_file.seek(0)
        for _ in range(self.skip_games):
            chess.pgn.skip_game(self.pgn_file)
        game = chess.pgn.read_game(self.pgn_file)
        current_game_idx = 0
        while game is not None:
            if self.limit_games is not None and current_game_idx >= self.limit_games:
                return
            for position in self._game_to_position_iter(game):
                if self.take_position_prob < 1 and random.random() > self.take_position_prob:
                    continue
                yield position
            game = chess.pgn.read_game(self.pgn_file)
            current_game_idx += 1


class LastMoveDataset(IterableDataset):
    def __init__(self, pgn_file, skip_games=0, limit_games=None, take_position_prob=1):
        self.position_dataset = PositionDataset(pgn_file, skip_games, limit_games, take_position_prob)

    def __iter__(self):
        for position_dict in iter(self.position_dataset):
            last_move_dest = position_dict['last_move'].to_square
            position = position_dict['position']
            yield (position_to_tensor(position), F.one_hot(torch.tensor(last_move_dest), 64).type(torch.float32))
