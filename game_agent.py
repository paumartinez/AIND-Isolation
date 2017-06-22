"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    return heuristica4(game, player)



def heuristica1(game, player):
    # First heuristic. 
    # This function evaluates the difference between the amount of legal moves available for the player and its oppnent.
    # If the player already won the game the function return +inf and if the player already losses it returns -inf.

    # If game already won
    if game.is_winner(player):
        return float("inf")

    # If gameover
    if game.is_loser(player):
        return float("-inf")

    # Amount of legal moves available
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    valor_h1 = float(player_moves - opponent_moves)

    return valor_h1



def heuristica2(game, player):

    # Second heuristic. 
    # This function evaluates the difference between the amount of legal moves available and for the player and its opponent, 
    # plus the difference between the moves available for each one after (the sum of moves available on each ramification of legal moves available).
    # If the player already won the game the function return +inf and if the player already losses it returns -inf.
    
    # If game already won
    if game.is_winner(player):
        return float("inf")

    # If gameover
    if game.is_loser(player):
        return float("-inf")

    # Legal moves available
    aux_player_moves = game.get_legal_moves(player)
    aux_opponent_moves = game.get_legal_moves(game.get_opponent(player))

    # Amount of legal moves available
    player_moves = len(aux_player_moves)
    opponent_moves = len(aux_opponent_moves)

    # Moves available of the player and its opponent
    player_availablemoves = float(sum([len(game.__get_moves__(move)) for move in aux_player_moves ]))
    opponent_availablemoves = float(sum([len(game.__get_moves__(move)) for move in aux_opponent_moves ]))
    
    # Value oh the second heuristic
    valor_h2 = float(player_moves + player_availablemoves - opponent_moves - opponent_availablemoves)
    
    return valor_h2




def heuristica3(game, player):
    # Third heuristic. 
    # This function evaluates the difference between the amount of legal moves available for the player and its oppnent,
    # plus a penalty if the position available is in a edge of the board (because in general is more dangerous to be
    # in an edge than in the middle of the board)
    # If the player already won the game the function return +inf and if the player already losses it returns -inf.

    # If game already won
    if game.is_winner(player):
        return float("inf")

    # If gameover
    if game.is_loser(player):
        return float("-inf")

    # Legal moves available
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    
  
    # Edges of the board
    edges = [(0,0)]

    baux = range((game.width-1))
    aaux = range((game.height-1))


    for a in aaux:
        edges.append((a,0))
        edges.append((a,(game.width-1)))

    for b in baux:
        edges.append((0,b))
        edges.append(((game.height-1),b))

    
    # advanced_game ponderates higher if the game is advanced, because it is more dangerous to be at an edge if there are fewer blank spaces
    advanced_game = 0.5
    if len(game.get_blank_spaces()) < game.width * game.height / 4:
        advanced_game = 1
    
    
    player_edges = [move for move in player_moves if move in edges]
    opponent_edges = [move for move in opponent_moves if move in edges]
    
    # Value oh the third heuristic
    valor_h3 = float(len(player_moves) - len(opponent_moves) + advanced_game * (len(opponent_edges) - len(player_edges)))

    return valor_h3

def heuristica4(game, player):
    # Third heuristic. 
    # This function evaluates the difference between the amount of legal moves available for the player and its oppnent,
    # plus a penalty if the position available is in a corner of the board (because in general is more dangerous to be
    # in a corner than in the middle of the board)
    # If the player already won the game the function return +inf and if the player already losses it returns -inf.

    # If game already won
    if game.is_winner(player):
        return float("inf")

    # If gameover
    if game.is_loser(player):
        return float("-inf")

    # Legal moves available
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    
    # Corners
    corners = [(0, 0), (0, (game.width - 1)), ((game.height - 1), 0), ((game.height - 1), (game.width - 1))]

    
    # advanced_game ponderates higher if the game is advanced, because it is more dangerous to be at corner if there are fewer blank spaces
    advanced_game = 0.5
    if len(game.get_blank_spaces()) < game.width * game.height / 4:
        advanced_game = 1
    

    player_corner = [move for move in player_moves if move in corners]
    opponent_corner = [move for move in opponent_moves if move in corners]
    
    # Value oh the fourth heuristic
    valor_h4 = float(len(player_moves) - len(opponent_moves) + advanced_game * (len(opponent_corner) - len(player_corner)))
    
    return valor_h4


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if len(legal_moves) == 0:
            return (-1,-1)

        # If first move, pick center position.
        if game.move_count == 0:
            return(int(game.height/2), int(game.width/2))

        last_move = (-1,-1)


        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            if self.iterative:
                aux_depth = 1

                if self.method == "minimax":

                    while True:
                        last_value, last_move = self.minimax(game, aux_depth)
                        if last_value == float("inf") or last_value == float("-inf"):
                            break
                        aux_depth += 1

                if self.method == "alphabeta":

                    while True:
                        last_value, last_move = self.alphabeta(game, aux_depth)
                        if last_value == float("inf") or last_value == float("-inf"):
                            break
                        aux_depth += 1

            else:
                if self.method == "minimax":
                    valor, last_move = self.minimax(game, self.search_depth)
                if self.method == "alphabeta":
                    valor, last_move = self.alphabeta(game, self.search_depth)


        except Timeout:
            # Handle any actions required at timeout, if necessary
            return last_move
            pass




        # Return the best move from the last completed search iteration
        return last_move


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
  

        # Possible legal moves for player
        legal_moves = game.get_legal_moves(game.active_player)

        # Stop conditions
        if depth == 0:
            return self.score(game,self), (-1,-1)

        if len(legal_moves) == 0:
            return self.score(game,self), (-1,-1)


        # Set Move improved
        move_imp = (-1,-1)

        # Set old_value
        if maximizing_player:
            old_value = float("-inf")
        else:
            old_value = float("inf")

        # Recursive minimax
        for move in legal_moves:

            new_value, move1 = self.minimax(game.forecast_move(move), depth-1, not maximizing_player)

            # Update variables
            if maximizing_player:
                if new_value > old_value:
                    old_value = new_value
                    move_imp = move    
            else:
                if new_value < old_value:
                    old_value = new_value
                    move_imp = move 
        
        # Return of the function Minimax
        return old_value, move_imp



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()


        # Possible legal moves for player
        legal_moves = game.get_legal_moves(game.active_player)

        # Stop conditions
        if depth == 0:
            return self.score(game,self), (-1,-1)

        if len(legal_moves) == 0:
            return self.score(game,self), (-1,-1)


        # Set Move improved
        move_imp = (-1,-1)

        # Set old_value
        if maximizing_player:
            old_value = float("-inf")
        else:
            old_value = float("inf")

        # Recursive Alphbeta
        for move in legal_moves:

            new_value, move1 = self.alphabeta(game.forecast_move(move), depth-1, alpha, beta, not maximizing_player)

            # Update variables
            if maximizing_player:
                if new_value > old_value:
                    old_value = new_value
                    move_imp = move    

                # Prune next node?
                if old_value >= beta:
                    return old_value, move_imp
                alpha = max(alpha,old_value)


            else:
                if new_value < old_value:
                    old_value = new_value
                    move_imp = move 

                if old_value <= alpha:
                    return old_value, move_imp
                beta = min(beta,old_value)    

        # Return of the function Alphabeta
        return old_value, move_imp

