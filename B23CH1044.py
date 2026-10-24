# B23CH1044.py
# Author: B23CH1044
# Competitive chess AI optimized for tournament wins (190 lines)

import math
import random
import time
from config import *
from board import Move

INF = 100000

class B23CH1044:
    def __init__(self, engine):
        self.engine = engine
        self.nodes_expanded = 0
        self.max_depth = 6
        self.time_limit = 1.4  
        self.start_time = 0.0
        self.depth = self.max_depth
        self.killer_moves = {}
        self.used_time = 0.0  # Track accumulated time usage
    
    def get_best_move(self):
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        self.nodes_expanded = 0
        self.start_time = time.perf_counter()
        best_move = None
        
        # Dynamic depth adjustment based on used_time
        if self.used_time <= 40:
            self.depth = 5
        elif self.used_time <= 50:
            self.depth = 4
        elif self.used_time <= 57:
            self.depth = 3
        else:
            self.depth = 1
        
        # Iterative deepening
        for depth in range(1, self.depth + 1):
            if self._time_exceeded():
                break
            try:
                move, score = self._search_root(depth, legal_moves)
                if move and not self._time_exceeded():
                    best_move = move
                    if abs(score) > 90000:  # Mate found
                        break
            except:
                break
        
        # Update used_time after move calculation
        move_time = time.perf_counter() - self.start_time
        self.used_time += move_time
        
        return best_move if best_move else self._emergency_move(legal_moves)
    
    def _search_root(self, depth, moves):
        best_move = None
        ordered_moves = self._order_moves(moves, depth)
        maximizing = self.engine.white_to_move
        
        if maximizing:
            best_score = -INF
            alpha = -INF
            for move in ordered_moves:
                if self._time_exceeded():
                    break
                self.engine.make_move(move)
                score = self._minimax(depth - 1, alpha, INF, False)
                self.engine.undo_move()
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
        else:
            best_score = INF
            beta = INF
            for move in ordered_moves:
                if self._time_exceeded():
                    break
                self.engine.make_move(move)
                score = self._minimax(depth - 1, -INF, beta, True)
                self.engine.undo_move()
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
        
        return best_move, best_score if best_move else (None, 0)
    
    def _minimax(self, depth, alpha, beta, maximizing):
        if self._time_exceeded():
            return self._quick_eval()
        
        self.nodes_expanded += 1
        game_state = self.engine.get_game_state()
        
        if depth <= 0 or game_state != "ongoing":
            if depth <= 0 and game_state == "ongoing":
                return self._quiescence(alpha, beta, maximizing, 2)
            return self.evaluate_board()
        
        moves = self.engine.get_legal_moves()
        if not moves:
            return self.evaluate_board()
        
        ordered_moves = self._order_moves(moves, depth)
        
        if maximizing:
            max_eval = -INF
            for move in ordered_moves:
                if self._time_exceeded():
                    break
                self.engine.make_move(move)
                eval_score = self._minimax(depth - 1, alpha, beta, False)
                self.engine.undo_move()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self._add_killer(move, depth)
                    break
            return max_eval
        else:
            min_eval = INF
            for move in ordered_moves:
                if self._time_exceeded():
                    break
                self.engine.make_move(move)
                eval_score = self._minimax(depth - 1, alpha, beta, True)
                self.engine.undo_move()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    self._add_killer(move, depth)
                    break
            return min_eval
    
    def _quiescence(self, alpha, beta, maximizing, depth):
        if depth <= 0 or self._time_exceeded():
            return self.evaluate_board()
        
        stand_pat = self.evaluate_board()
        if maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            captures = [m for m in self.engine.get_legal_moves() if m.piece_captured != EMPTY_SQUARE]
            for move in sorted(captures, key=lambda m: abs(PIECE_VALUES.get(m.piece_captured, 0)), reverse=True):
                self.engine.make_move(move)
                score = self._quiescence(alpha, beta, False, depth - 1)
                self.engine.undo_move()
                if score >= beta:
                    return beta
                alpha = max(alpha, score)
            return alpha
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
            captures = [m for m in self.engine.get_legal_moves() if m.piece_captured != EMPTY_SQUARE]
            for move in sorted(captures, key=lambda m: abs(PIECE_VALUES.get(m.piece_captured, 0)), reverse=True):
                self.engine.make_move(move)
                score = self._quiescence(alpha, beta, True, depth - 1)
                self.engine.undo_move()
                if score <= alpha:
                    return alpha
                beta = min(beta, score)
            return beta
    
    def _order_moves(self, moves, depth):
        def score_move(move):
            score = 0
            # Captures first (MVV-LVA)
            if move.piece_captured != EMPTY_SQUARE:
                victim = abs(PIECE_VALUES.get(move.piece_captured, 0))
                attacker = abs(PIECE_VALUES.get(move.piece_moved, 0))
                score += 10000 + victim - attacker // 10
            
            # Killer moves
            if depth in self.killer_moves:
                if any(self._same_move(move, km) for km in self.killer_moves[depth]):
                    score += 5000
            
            # Positional bonuses
            piece_type = move.piece_moved[1] if len(move.piece_moved) >= 2 else ''
            row, col = move.end_row, move.end_col
            
            if piece_type == 'P':
                score += PAWN_PST[row][col] * 2
            elif piece_type == 'N':
                score += KNIGHT_PST[row][col] * 3
            elif piece_type == 'B':
                score += BISHOP_PST[row][col] * 2
            elif piece_type == 'K':
                score += KING_PST_LATE_GAME[row][col]
            
            # Center control
            if row in [3, 4] and col in [1, 2]:
                score += 100
            
            return score
        
        return sorted(moves, key=score_move, reverse=True)
    
    def evaluate_board(self):
        game_state = self.engine.get_game_state()
        if game_state == "checkmate":
            return -99999 if self.engine.white_to_move else 99999
        if game_state == "stalemate":
            return 0
        
        score = 0
        
        # Material and positional evaluation
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                piece = self.engine.board[row][col]
                if piece == EMPTY_SQUARE:
                    continue
                
                # Material
                score += PIECE_VALUES.get(piece, 0)
                
                # Positional
                color, piece_type = piece[0], piece[1]
                pst_val = 0
                
                if piece_type == 'P':
                    pst_val = PAWN_PST[row][col]
                elif piece_type == 'N':
                    pst_val = KNIGHT_PST[row][col]
                elif piece_type == 'B':
                    pst_val = BISHOP_PST[row][col]
                elif piece_type == 'K':
                    pst_val = KING_PST_LATE_GAME[row][col]
                
                if color == 'w':
                    score += pst_val * 1.5
                else:
                    mirror_row = BOARD_HEIGHT - 1 - row
                    if piece_type == 'P':
                        score -= PAWN_PST[mirror_row][col] * 1.5
                    elif piece_type == 'N':
                        score -= KNIGHT_PST[mirror_row][col] * 1.5
                    elif piece_type == 'B':
                        score -= BISHOP_PST[mirror_row][col] * 1.5
                    elif piece_type == 'K':
                        score -= KING_PST_LATE_GAME[mirror_row][col] * 1.5
        
        # Mobility evaluation
        current_moves = len(self.engine.get_legal_moves())
        self.engine.white_to_move = not self.engine.white_to_move
        opp_moves = len(self.engine.get_legal_moves())
        self.engine.white_to_move = not self.engine.white_to_move
        
        mobility_diff = current_moves - opp_moves
        if self.engine.white_to_move:
            score += mobility_diff * 8
        else:
            score -= mobility_diff * 8
        
        # Check penalty
        if self.engine.is_in_check():
            penalty = 60
            score += penalty if not self.engine.white_to_move else -penalty
        
        return score
    
    def _emergency_move(self, moves):
        # Priority: checkmate > best capture > check > center > random
        for move in moves:
            self.engine.make_move(move)
            if self.engine.get_game_state() == "checkmate":
                self.engine.undo_move()
                return move
            self.engine.undo_move()
        
        captures = [m for m in moves if m.piece_captured != EMPTY_SQUARE]
        if captures:
            return max(captures, key=lambda m: abs(PIECE_VALUES.get(m.piece_captured, 0)))
        
        for move in moves:
            self.engine.make_move(move)
            is_check = self.engine.is_in_check()
            self.engine.undo_move()
            if is_check:
                return move
        
        center_moves = [m for m in moves if m.end_row in [3, 4] and m.end_col in [1, 2]]
        return random.choice(center_moves) if center_moves else random.choice(moves)
    
    def _quick_eval(self):
        score = 0
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                piece = self.engine.board[row][col]
                if piece != EMPTY_SQUARE:
                    score += PIECE_VALUES.get(piece, 0)
        return score
    
    def _time_exceeded(self):
        return (time.perf_counter() - self.start_time) >= (self.time_limit - 0.05)
    
    def _add_killer(self, move, depth):
        if depth not in self.killer_moves:
            self.killer_moves[depth] = []
        if len(self.killer_moves[depth]) >= 2:
            self.killer_moves[depth].pop(0)
        self.killer_moves[depth].append(move)
    
    def _same_move(self, m1, m2):
        return (m1.start_row == m2.start_row and m1.start_col == m2.start_col and
                m1.end_row == m2.end_row and m1.end_col == m2.end_col)