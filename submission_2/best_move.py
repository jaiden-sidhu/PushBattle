from PushBattle import Game, PLAYER1, PLAYER2, EMPTY
import numpy as np

class Minimax:

    def get_best_move(self, game):
        board = game.board
        player = game.current_player
        bestScore = float('-inf')
        bestRow = 0
        bestCol = 0

        isMovement = False
        if (game.current_player == PLAYER1):
            if (game.p1_pieces >= 8):
                isMovement = True
        
        if (game.current_player == PLAYER2):
            if (game.p2_pieces >= 8):
                isMovement = True

        if (not isMovement):
            for i in range(8):
                for j in range(8):
                    if (game.is_valid_placement(i, j)):
                        p1_pieces = game.p1_pieces
                        p2_pieces = game.p2_pieces

                        scoreOfMove = self.minimax_placement(game, i, j, 5, float('-inf'), float('inf'), player, player)

                        game.p1_pieces = p1_pieces
                        game.p2_pieces = p2_pieces
                        print("testing", i, j, ":", scoreOfMove)
                        if (scoreOfMove > bestScore):
                            bestScore = scoreOfMove
                            bestRow = i
                            bestCol = j
            move = [bestRow, bestCol]
            # print(move)
        else:
            bestMoveRow = 0
            bestMoveCol = 0
            for i in range(8):
                for j in range(8):
                    if board[i][j] == player:
                        for x in range(8):
                            for y in range(8):
                                if game.is_valid_move(i, j, x, y):
                                    p1_pieces = game.p1_pieces
                                    p2_pieces = game.p2_pieces
                                    scoreOfMove = self.minimax_movement(game, i, j, x, y, 5, float('-inf'), float('inf'), player, player)
                                    game.p1_pieces = p1_pieces
                                    game.p2_pieces = p2_pieces
                                    if (scoreOfMove > bestScore):
                                        bestScore = scoreOfMove
                                        bestRow = i
                                        bestCol = j
                                        bestMoveRow = x
                                        bestMoveCol = y
            move = [bestRow, bestCol, bestMoveRow, bestMoveCol]
            print(move)
        
        return move

# call w/ minimax_placement([game], [row], [col], 5, float('inf'), float(inf))
    def minimax_placement(self, game, row, col, depth, alpha, beta, currPlayer, targetPlayer):
        if depth == 0:
            return 0
        
        newGame = game

        newGame.current_player = currPlayer
        newGame.place_checker(row, col)

        winner = newGame.check_winner()
        if winner == targetPlayer:
            return ((5-depth) ^ 64)
        elif winner != EMPTY:
            return -1 * ((5-depth) ^ 64)
        
        if currPlayer == PLAYER1:
            currPlayer = PLAYER2
        else:
            currPlayer = PLAYER1

        totalScore = 0
        if currPlayer == targetPlayer:
            best_score = float('-inf') 
            for i in range(8):
                for j in range(8):
                    if newGame.is_valid_placement(i, j):
                        p1_pieces = game.p1_pieces
                        p2_pieces = game.p2_pieces
                        eval_score = self.minimax_placement(newGame, i, j, (depth - 1), alpha, beta, currPlayer, targetPlayer)
                        game.p1_pieces = p1_pieces
                        game.p2_pieces = p2_pieces
                        
                        totalScore += eval_score
                        best_score = max(best_score, totalScore)
                        alpha = max(alpha, best_score)
                        if beta <= alpha:
                            break
        else:
            best_score = float('inf')
            for i in range(8):
                for j in range(8):
                    if newGame.is_valid_placement(i, j):
                        p1_pieces = game.p1_pieces
                        p2_pieces = game.p2_pieces
                        eval_score = self.minimax_placement(newGame, i, j, (depth - 1), alpha, beta, currPlayer, targetPlayer)
                        game.p1_pieces = p1_pieces
                        game.p2_pieces = p2_pieces
                        totalScore += eval_score
                        best_score = min(best_score, totalScore)
                        beta = min(beta, best_score)
                        if beta <= alpha:
                            break
        
        return totalScore

    # call w/ minimax_placement([game], [init_row], [init_col], [init_nrow], [init_ncol], 5, float('inf'), float(inf))
    def minimax_movement(self, game, init_row, init_col, new_row, new_col, depth, alpha, beta, currPlayer, targetPlayer):
        if depth == 0:
            return 0
        
        newGame = game

        newGame.current_player = currPlayer
        newGame.move_checker(init_row, init_col, new_row, new_col)

        winner = newGame.check_winner()
        if winner == targetPlayer:
            return ((5-depth) ^ 64)
        elif winner != EMPTY:
            return -1 * ((5-depth) ^ 64)
        
        if currPlayer == PLAYER1:
            currPlayer = PLAYER2
        else:
            currPlayer = PLAYER1

        if currPlayer == targetPlayer:
            best_score = -float('inf') 
            for i in range(8):
                for j in range(8):
                    if newGame.is_valid_move(new_row, new_col, i, j):
                        p1_pieces = game.p1_pieces
                        p2_pieces = game.p2_pieces
                        eval_score = self.minimax_movement(newGame, new_row, new_col, i, j, (depth - 1), alpha, beta, currPlayer, targetPlayer)
                        game.p1_pieces = p1_pieces
                        game.p2_pieces = p2_pieces
                        best_score = max(best_score, eval_score)
                        alpha = max(alpha, best_score)
                        if beta <= alpha:
                            break
        else:
            best_score = float('inf')
            for i in range(8):
                for j in range(8):
                    if newGame.is_valid_move(new_row, new_col, i, j):
                        p1_pieces = game.p1_pieces
                        p2_pieces = game.p2_pieces
                        eval_score = self.minimax_movement(newGame, new_row, new_col, i, j, (depth - 1), alpha, beta, currPlayer, targetPlayer)
                        game.p1_pieces = p1_pieces
                        game.p2_pieces = p2_pieces
                        best_score = min(best_score, eval_score)
                        beta = min(beta, best_score)
                        if beta <= alpha:
                            break

        return best_score